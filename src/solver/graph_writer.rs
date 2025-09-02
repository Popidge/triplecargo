use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{self, BufWriter, Write};

/// Default buffer size for BufWriter in graph JSONL sinks (32 MiB)
pub const BUF_WRITER_CAP_BYTES: usize = 32 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct GraphSinkStats {
    pub total_lines: u64,
    pub frame_count: u64,
    pub nodes_sha256_hex: String,
    pub index_sha256_hex: Option<String>,
}

pub trait GraphJsonlSink {
    fn write_line(&mut self, json_line: &[u8], turn: u8) -> io::Result<()>;
    fn finish_mut(&mut self) -> io::Result<GraphSinkStats>;
}

/*************** Plain JSONL writer (uncompressed) ***************/
pub struct PlainJsonlWriter {
    nodes: BufWriter<File>,
    hasher_nodes: Sha256,
    total_lines: u64,
    sync_final: bool,
}

impl PlainJsonlWriter {
    pub fn new(nodes_file: File, buf_bytes: usize, sync_final: bool) -> Self {
        Self {
            nodes: BufWriter::with_capacity(buf_bytes.max(1024 * 1024), nodes_file),
            hasher_nodes: Sha256::new(),
            total_lines: 0,
            sync_final,
        }
    }
}

impl GraphJsonlSink for PlainJsonlWriter {
    fn write_line(&mut self, json_line: &[u8], _turn: u8) -> io::Result<()> {
        self.nodes.write_all(json_line)?;
        self.nodes.write_all(b"\n")?;
        self.hasher_nodes.update(json_line);
        self.hasher_nodes.update(b"\n");
        self.total_lines = self.total_lines.saturating_add(1);
        Ok(())
    }

    fn finish_mut(&mut self) -> io::Result<GraphSinkStats> {
        // Single-flush policy: flush once at the end
        self.nodes.flush()?;
        // Optional final fsync for crash safety
        if self.sync_final {
            // Safe: BufWriter holds a File
            self.nodes.get_ref().sync_all()?;
        }
        let digest = std::mem::take(&mut self.hasher_nodes).finalize();
        Ok(GraphSinkStats {
            total_lines: self.total_lines,
            frame_count: 0,
            nodes_sha256_hex: hex::encode(digest),
            index_sha256_hex: None,
        })
    }
}

/*************** Zstd frames JSONL writer (compressed) ***************/

// Helper Write impl that forwards into a BufWriter<File>, updates a SHA256, and tracks total bytes written.
struct HashingFileWrite<'a> {
    inner: &'a mut BufWriter<File>,
    hasher: &'a mut Sha256,
    pos: &'a mut u64,
}

impl<'a> Write for HashingFileWrite<'a> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.hasher.update(&buf[..n]);
        *self.pos = (*self.pos).saturating_add(n as u64);
        Ok(n)
    }
    fn flush(&mut self) -> io::Result<()> {
        // Defer outer BufWriter flushing to sink finalize for single-flush policy
        Ok(())
    }
}

pub struct ZstdFramesJsonlWriter {
    nodes: BufWriter<File>,
    idx: Option<BufWriter<File>>,

    hasher_nodes: Sha256,
    hasher_idx: Option<Sha256>,

    // Tracks total compressed bytes written so far (for index offsets).
    compressed_pos: u64,

    // Accumulator for one frame worth of JSONL (uncompressed)
    frame_buf: Vec<u8>,
    frame_lines_target: usize,
    frame_max_bytes: usize,
    frame_lines_count: usize,

    // Running counters
    total_lines: u64,
    frame_count: u64,
    global_line_cursor: u64, // next line number to assign as line_start for a new frame

    // Per-frame stats
    cur_turn_min: Option<u8>,
    cur_turn_max: Option<u8>,

    // Compression config
    zstd_level: i32,
    zstd_threads: usize,

    // Final fsync policy
    sync_final: bool,
}

impl ZstdFramesJsonlWriter {
    pub fn new(
        nodes_file: File,
        idx_file: Option<File>,
        buf_bytes: usize,
        zstd_level: i32,
        zstd_threads: usize,
        frame_lines_target: usize,
        frame_max_bytes: usize,
        sync_final: bool,
    ) -> Self {
        let idx = idx_file.map(|f| BufWriter::with_capacity(buf_bytes.max(1024 * 1024), f));
        let hasher_idx = idx.as_ref().map(|_| Sha256::new());
        Self {
            nodes: BufWriter::with_capacity(buf_bytes.max(8 * 1024 * 1024), nodes_file),
            idx,
            hasher_nodes: Sha256::new(),
            hasher_idx,
            compressed_pos: 0,
            // Reserve for large frames: at least 128 MiB or requested frame_max_bytes
            frame_buf: Vec::with_capacity(frame_max_bytes.max(128 * 1024 * 1024)),
            frame_lines_target: frame_lines_target.max(1),
            frame_max_bytes: frame_max_bytes.max(1 * 1024 * 1024),
            frame_lines_count: 0,
            total_lines: 0,
            frame_count: 0,
            global_line_cursor: 0,
            cur_turn_min: None,
            cur_turn_max: None,
            zstd_level,
            zstd_threads: zstd_threads.max(1),
            sync_final,
        }
    }

    fn flush_frame(&mut self) -> io::Result<()> {
        if self.frame_lines_count == 0 {
            return Ok(());
        }

        // Compressed byte offset at start of this new frame (tracked internally; no pre-flush)
        let offset = self.compressed_pos;

        // Create a zstd encoder that writes compressed bytes directly to nodes,
        // while hashing those bytes for nodes SHA256, and updates compressed_pos.
        {
            let mut hashing_writer = HashingFileWrite {
                inner: &mut self.nodes,
                hasher: &mut self.hasher_nodes,
                pos: &mut self.compressed_pos,
            };
            let mut encoder =
                zstd::stream::write::Encoder::new(&mut hashing_writer, self.zstd_level)?;
            // Attempt to enable multithreading (zstd crate compiled with `zstdmt` feature). Ignore errors if not supported.
            let _ = encoder.multithread(self.zstd_threads as u32);
            // We don't set content size as it's streaming/unknown
            encoder.write_all(&self.frame_buf)?;
            // Finish to flush frame footer into the inner writer (position updated by HashingFileWrite)
            let _inner = encoder.finish();
            // hashing_writer drops here; nodes remains borrowed mutably within self
        }

        // Write one index line if enabled
        if let (Some(idx), Some(h)) = (self.idx.as_mut(), self.hasher_idx.as_mut()) {
            // Safe unwraps: we only flush when we have at least one line
            let turn_min = self.cur_turn_min.unwrap_or(0);
            let turn_max = self.cur_turn_max.unwrap_or(turn_min);
            let line = serde_json::json!({
                "frame": self.frame_count,
                "line_start": self.global_line_cursor,
                "byte_offset": offset,
                "lines": self.frame_lines_count as u64,
                "turn_min": turn_min,
                "turn_max": turn_max
            });
            let mut tmp =
                serde_json::to_vec(&line).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
            tmp.push(b'\n');
            idx.write_all(&tmp)?;
            h.update(&tmp);
        }

        self.frame_count = self.frame_count.saturating_add(1);
        self.total_lines = self.total_lines.saturating_add(self.frame_lines_count as u64);
        self.global_line_cursor = self
            .global_line_cursor
            .saturating_add(self.frame_lines_count as u64);

        // Reset for next frame
        self.frame_lines_count = 0;
        self.frame_buf.clear();
        self.cur_turn_min = None;
        self.cur_turn_max = None;

        Ok(())
        }
}

impl GraphJsonlSink for ZstdFramesJsonlWriter {
    fn write_line(&mut self, json_line: &[u8], turn: u8) -> io::Result<()> {
        // Append line and newline to frame buffer
        self.frame_buf.extend_from_slice(json_line);
        self.frame_buf.push(b'\n');

        // Track per-frame metadata
        self.frame_lines_count = self.frame_lines_count.saturating_add(1);
        match (self.cur_turn_min, self.cur_turn_max) {
            (None, None) => {
                self.cur_turn_min = Some(turn);
                self.cur_turn_max = Some(turn);
            }
            (Some(lo), Some(hi)) => {
                if turn < lo {
                    self.cur_turn_min = Some(turn);
                }
                if turn > hi {
                    self.cur_turn_max = Some(turn);
                }
            }
            _ => {
                self.cur_turn_min = Some(turn);
                self.cur_turn_max = Some(turn);
            }
        }

        // Check flush conditions
        if self.frame_lines_count >= self.frame_lines_target || self.frame_buf.len() >= self.frame_max_bytes {
            self.flush_frame()?;
        }

        Ok(())
    }

    fn finish_mut(&mut self) -> io::Result<GraphSinkStats> {
        // Flush any partial frame
        self.flush_frame()?;
        // Single-flush policy: flush once at the end
        self.nodes.flush()?;
        if let Some(idx) = self.idx.as_mut() {
            idx.flush()?;
        }
        // Optional final fsync for crash safety (Linux target)
        if self.sync_final {
            self.nodes.get_ref().sync_all()?;
            if let Some(idx_ref) = self.idx.as_ref() {
                idx_ref.get_ref().sync_all()?;
            }
        }

        let nodes_digest = std::mem::take(&mut self.hasher_nodes).finalize();
        let index_digest = self.hasher_idx.take().map(|h| hex::encode(h.finalize()));

        Ok(GraphSinkStats {
            total_lines: self.total_lines,
            frame_count: self.frame_count,
            nodes_sha256_hex: hex::encode(nodes_digest),
            index_sha256_hex: index_digest,
        })
    }
}