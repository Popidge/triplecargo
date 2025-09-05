use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::thread;

use crossbeam_channel::{bounded, Sender, Receiver};
use std::collections::BTreeMap;

/// Default buffer size for BufWriter in graph JSONL sinks (32 MiB)
pub const BUF_WRITER_CAP_BYTES: usize = 32 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct GraphSinkStats {
    pub total_lines: u64,
    pub frame_count: u64,
    // When hashing is enabled this will contain the hex digest; when disabled it will be None.
    pub nodes_sha256_hex: Option<String>,
    pub index_sha256_hex: Option<String>,
    // When sharding is used, provide per-shard node file digests (in shard index order).
    pub nodes_sha256_list: Option<Vec<String>>,
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
    hashing: bool,
}

impl PlainJsonlWriter {
    pub fn new(nodes_file: File, buf_bytes: usize, sync_final: bool, hashing: bool) -> Self {
        Self {
            nodes: BufWriter::with_capacity(buf_bytes.max(1024 * 1024), nodes_file),
            hasher_nodes: Sha256::new(),
            total_lines: 0,
            sync_final,
            hashing,
        }
    }
}

impl GraphJsonlSink for PlainJsonlWriter {
    fn write_line(&mut self, json_line: &[u8], _turn: u8) -> io::Result<()> {
        self.nodes.write_all(json_line)?;
        self.nodes.write_all(b"\n")?;
        if self.hashing {
            self.hasher_nodes.update(json_line);
            self.hasher_nodes.update(b"\n");
        }
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
        let nodes_sha = if self.hashing {
            let digest = std::mem::take(&mut self.hasher_nodes).finalize();
            Some(hex::encode(digest))
        } else {
            None
        };
        Ok(GraphSinkStats {
            total_lines: self.total_lines,
            frame_count: 0,
            nodes_sha256_hex: nodes_sha,
            index_sha256_hex: None,
            nodes_sha256_list: None,
        })
    }
}

/*************** Zstd frames JSONL writer (compressed) ***************/

// Helper Write impl that forwards into a BufWriter<File>, updates a SHA256, and tracks total bytes written.
struct HashingFileWrite<'a> {
    inner: &'a mut BufWriter<File>,
    hasher: &'a mut Sha256,
    pos: &'a mut u64,
    hashing: bool,
}

impl<'a> Write for HashingFileWrite<'a> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = self.inner.write(buf)?;
        if self.hashing {
            self.hasher.update(&buf[..n]);
        }
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
 
    // Whether to compute/emit SHA256 digests
    hashing: bool,
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
        hashing: bool,
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
            hashing,
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
                hashing: self.hashing,
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
            if self.hashing {
                h.update(&tmp);
            }
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
            nodes_sha256_hex: Some(hex::encode(nodes_digest)),
            index_sha256_hex: index_digest,
            nodes_sha256_list: None,
        })
    }
}

// ================= Async ZSTD frames writer (producer/consumer with optional compression pool) =================

#[derive(Debug)]
struct Frame {
    id: u64,
    uncompressed: Vec<u8>,
    turn_min: u8,
    turn_max: u8,
    line_start: u64,
    line_count: u32,
}

enum FrameMsg {
    Data(Frame),
    End,
}

#[derive(Debug)]
struct CompressedFrame {
    id: u64,
    buf: Vec<u8>,
    turn_min: u8,
    turn_max: u8,
    line_start: u64,
    line_count: u32,
}

enum CompMsg {
    Data(CompressedFrame),
    End,
}

struct WriterOutcome {
    total_lines: u64,
    frame_count: u64,
    nodes_sha256_hex: String,
    index_sha256_hex: Option<String>,
}

/// Asynchronous ZSTD frames writer with optional 2-stage pipeline:
/// - Producer (this object) accumulates JSONL into frames and enqueues them into a bounded channel
/// - If zstd_workers == 0: a single writer thread compresses and writes frames in order
/// - If zstd_workers > 0: a pool compresses into memory, then a single writer emits compressed frames sequentially
pub struct AsyncZstdFramesJsonlWriter {
    // producer-side frame builder
    frame_buf: Vec<u8>,
    frame_lines_target: usize,
    frame_max_bytes: usize,
    frame_lines_count: usize,
    cur_turn_min: Option<u8>,
    cur_turn_max: Option<u8>,
    next_frame_id: u64,
    global_line_cursor: u64,

    // producer -> stage1 (raw frames)
    tx: Sender<FrameMsg>,

    // join handles
    writer_join: Option<thread::JoinHandle<io::Result<WriterOutcome>>>,
    comp_joins: Vec<thread::JoinHandle<()>>,

    // number of compression workers (0 = single-stage)
    workers: usize,
    // whether to compute/emit SHA digests
    hashing: bool,
}

impl AsyncZstdFramesJsonlWriter {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        nodes_file: File,
        idx_file: Option<File>,
        buf_bytes: usize,
        zstd_level: i32,
        zstd_threads: usize,
        frame_lines_target: usize,
        frame_max_bytes: usize,
        writer_queue_frames: usize,
        zstd_workers: usize,
        writer_queue_compressed: usize,
        sync_final: bool,
        hashing: bool,
    ) -> Self {
        let workers = zstd_workers;
        let (tx, raw_rx) = bounded::<FrameMsg>(writer_queue_frames.max(1));

        if workers == 0 {
            // Single-stage path: writer compresses and writes
            let join = thread::spawn(move || -> io::Result<WriterOutcome> {
                let mut nodes = BufWriter::with_capacity(buf_bytes.max(8 * 1024 * 1024), nodes_file);
                let mut idx = idx_file.map(|f| BufWriter::with_capacity(buf_bytes.max(1024 * 1024), f));
                let mut hasher_nodes = Sha256::new();
                let mut hasher_idx = idx.as_ref().map(|_| Sha256::new());
                let mut compressed_pos: u64 = 0;
                let mut total_lines: u64 = 0;
                let mut frame_count: u64 = 0;
                let hashing_local = hashing;

                while let Ok(msg) = raw_rx.recv() {
                    match msg {
                        FrameMsg::Data(f) => {
                            let offset = compressed_pos;
                            {
                                let mut counting_writer = HashingFileWrite {
                                    inner: &mut nodes,
                                    hasher: &mut hasher_nodes,
                                    pos: &mut compressed_pos,
                                    hashing: hashing_local,
                                };
                                let mut encoder = zstd::stream::write::Encoder::new(&mut counting_writer, zstd_level)?;
                                let _ = encoder.multithread(zstd_threads.max(1) as u32);
                                encoder.write_all(&f.uncompressed)?;
                                let _ = encoder.finish();
                            }
                            if let (Some(idxw), Some(h)) = (idx.as_mut(), hasher_idx.as_mut()) {
                                let line = serde_json::json!({
                                    "frame": frame_count,
                                    "line_start": f.line_start,
                                    "byte_offset": offset,
                                    "lines": f.line_count as u64,
                                    "turn_min": f.turn_min,
                                    "turn_max": f.turn_max
                                });
                                let mut tmp = serde_json::to_vec(&line).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
                                tmp.push(b'\n');
                                idxw.write_all(&tmp)?;
                                if hashing_local {
                                    h.update(&tmp);
                                }
                            }
                            frame_count = frame_count.saturating_add(1);
                            total_lines = total_lines.saturating_add(f.line_count as u64);
                        }
                        FrameMsg::End => break,
                    }
                }

                nodes.flush()?;
                if let Some(idxw) = idx.as_mut() {
                    idxw.flush()?;
                }
                if sync_final {
                    nodes.get_ref().sync_all()?;
                    if let Some(idxw) = idx.as_ref() {
                        idxw.get_ref().sync_all()?;
                    }
                }

                let nodes_sha = hex::encode(hasher_nodes.finalize());
                let index_sha = hasher_idx.map(|h| hex::encode(h.finalize()));
                Ok(WriterOutcome {
                    total_lines,
                    frame_count,
                    nodes_sha256_hex: nodes_sha,
                    index_sha256_hex: index_sha,
                })
            });

            Self {
                frame_buf: Vec::with_capacity(frame_max_bytes.max(128 * 1024 * 1024)),
                frame_lines_target: frame_lines_target.max(1),
                frame_max_bytes: frame_max_bytes.max(1 * 1024 * 1024),
                frame_lines_count: 0,
                cur_turn_min: None,
                cur_turn_max: None,
                next_frame_id: 0,
                global_line_cursor: 0,
                tx,
                writer_join: Some(join),
                comp_joins: Vec::new(),
                workers,
                hashing,
            }
        } else {
            // Two-stage path: compression pool -> single writer
            let (comp_tx, comp_rx) = bounded::<CompMsg>(writer_queue_compressed.max(1));

            // Compression workers
            let mut comp_joins: Vec<thread::JoinHandle<()>> = Vec::with_capacity(workers);
            for _wid in 0..workers {
                let raw_rx_c: Receiver<FrameMsg> = raw_rx.clone();
                let comp_tx_c = comp_tx.clone();
                let jl = thread::spawn(move || {
                    while let Ok(msg) = raw_rx_c.recv() {
                        match msg {
                            FrameMsg::Data(f) => {
                                // Compress into memory buffer
                                let mut dst: Vec<u8> = Vec::with_capacity(f.uncompressed.len() / 4 + 1024);
                                if let Ok(mut encoder) = zstd::stream::write::Encoder::new(&mut dst, zstd_level) {
                                    let _ = encoder.multithread(zstd_threads.max(1) as u32);
                                    let _ = encoder.write_all(&f.uncompressed);
                                    let _ = encoder.finish();
                                }
                                let _ = comp_tx_c.send(CompMsg::Data(CompressedFrame {
                                    id: f.id,
                                    buf: dst,
                                    turn_min: f.turn_min,
                                    turn_max: f.turn_max,
                                    line_start: f.line_start,
                                    line_count: f.line_count,
                                }));
                            }
                            FrameMsg::End => {
                                let _ = comp_tx_c.send(CompMsg::End);
                                break;
                            }
                        }
                    }
                });
                comp_joins.push(jl);
            }
            drop(comp_tx); // writer holds the last receiver, senders are in workers

            // Writer thread: preserve order by id
            let join = thread::spawn(move || -> io::Result<WriterOutcome> {
                let mut nodes = BufWriter::with_capacity(buf_bytes.max(8 * 1024 * 1024), nodes_file);
                let mut idx = idx_file.map(|f| BufWriter::with_capacity(buf_bytes.max(1024 * 1024), f));
                let mut hasher_nodes = Sha256::new();
                let mut hasher_idx = idx.as_ref().map(|_| Sha256::new());
                let mut compressed_pos: u64 = 0;
                let mut total_lines: u64 = 0;
                let mut frame_count: u64 = 0;

                let mut next_id: u64 = 0;
                let mut ends_seen: usize = 0;
                let mut pending: BTreeMap<u64, CompressedFrame> = BTreeMap::new();

                while ends_seen < workers || !pending.is_empty() {
                    match comp_rx.recv() {
                        Ok(CompMsg::Data(cf)) => {
                            pending.insert(cf.id, cf);
                            // Drain in-order frames
                            while let Some(cf) = pending.remove(&next_id) {
                                let offset = compressed_pos;
                                // Write pre-compressed bytes
                                nodes.write_all(&cf.buf)?;
                                hasher_nodes.update(&cf.buf);
                                compressed_pos = compressed_pos.saturating_add(cf.buf.len() as u64);

                                if let (Some(idxw), Some(h)) = (idx.as_mut(), hasher_idx.as_mut()) {
                                    let line = serde_json::json!({
                                        "frame": frame_count,
                                        "line_start": cf.line_start,
                                        "byte_offset": offset,
                                        "lines": cf.line_count as u64,
                                        "turn_min": cf.turn_min,
                                        "turn_max": cf.turn_max
                                    });
                                    let mut tmp = serde_json::to_vec(&line).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
                                    tmp.push(b'\n');
                                    idxw.write_all(&tmp)?;
                                    h.update(&tmp);
                                }

                                frame_count = frame_count.saturating_add(1);
                                total_lines = total_lines.saturating_add(cf.line_count as u64);
                                next_id = next_id.saturating_add(1);
                            }
                        }
                        Ok(CompMsg::End) => {
                            ends_seen = ends_seen.saturating_add(1);
                        }
                        Err(_) => break,
                    }
                }

                nodes.flush()?;
                if let Some(idxw) = idx.as_mut() {
                    idxw.flush()?;
                }
                if sync_final {
                    nodes.get_ref().sync_all()?;
                    if let Some(idxw) = idx.as_ref() {
                        idxw.get_ref().sync_all()?;
                    }
                }

                let nodes_sha = hex::encode(hasher_nodes.finalize());
                let index_sha = hasher_idx.map(|h| hex::encode(h.finalize()));
                Ok(WriterOutcome {
                    total_lines,
                    frame_count,
                    nodes_sha256_hex: nodes_sha,
                    index_sha256_hex: index_sha,
                })
            });

            Self {
                frame_buf: Vec::with_capacity(frame_max_bytes.max(128 * 1024 * 1024)),
                frame_lines_target: frame_lines_target.max(1),
                frame_max_bytes: frame_max_bytes.max(1 * 1024 * 1024),
                frame_lines_count: 0,
                cur_turn_min: None,
                cur_turn_max: None,
                next_frame_id: 0,
                global_line_cursor: 0,
                tx,
                writer_join: Some(join),
                comp_joins,
                workers,
                hashing,
            }
        }
    }

    fn emit_frame(&mut self) -> io::Result<()> {
        if self.frame_lines_count == 0 {
            return Ok(());
        }
        let turn_min = self.cur_turn_min.unwrap_or(0);
        let turn_max = self.cur_turn_max.unwrap_or(turn_min);
        let line_start = self.global_line_cursor;
        let line_count_u32 = self.frame_lines_count as u32;

        let buf = std::mem::replace(
            &mut self.frame_buf,
            Vec::with_capacity(self.frame_max_bytes.max(128 * 1024 * 1024)),
        );

        let frame = Frame {
            id: self.next_frame_id,
            uncompressed: buf,
            turn_min,
            turn_max,
            line_start,
            line_count: line_count_u32,
        };
        // advance producer counters deterministically
        self.next_frame_id = self.next_frame_id.saturating_add(1);
        self.global_line_cursor = self.global_line_cursor.saturating_add(self.frame_lines_count as u64);
        self.frame_lines_count = 0;
        self.cur_turn_min = None;
        self.cur_turn_max = None;

        // send (blocks when queue full)
        self.tx
            .send(FrameMsg::Data(frame))
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "writer channel closed"))
    }
}

impl GraphJsonlSink for AsyncZstdFramesJsonlWriter {
    fn write_line(&mut self, json_line: &[u8], turn: u8) -> io::Result<()> {
        self.frame_buf.extend_from_slice(json_line);
        self.frame_buf.push(b'\n');

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

        if self.frame_lines_count >= self.frame_lines_target || self.frame_buf.len() >= self.frame_max_bytes {
            self.emit_frame()?;
        }
        Ok(())
    }

    fn finish_mut(&mut self) -> io::Result<GraphSinkStats> {
        // flush last frame if pending
        self.emit_frame()?;

        // signal end(s)
        let ends = if self.workers == 0 { 1 } else { self.workers };
        for _ in 0..ends {
            self.tx
                .send(FrameMsg::End)
                .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "writer channel closed"))?;
        }

        // join compression workers (if any)
        for j in self.comp_joins.drain(..) {
            let _ = j.join();
        }

        // join writer and return stats
        let handle = self.writer_join.take().expect("writer join present");
        let outcome = handle.join().map_err(|_| {
            io::Error::new(io::ErrorKind::Other, "writer thread panicked")
        })??;

        Ok(GraphSinkStats {
            total_lines: outcome.total_lines,
            frame_count: outcome.frame_count,
            nodes_sha256_hex: Some(outcome.nodes_sha256_hex),
            index_sha256_hex: outcome.index_sha256_hex,
            nodes_sha256_list: None,
        })
    }
}
//
// Async sharded ZSTD frames writer (single-coordinator implementation)
//
// Coordinator receives completed Frames from the producer and deterministically
// assigns them to shards (shard = frame_id % shards). The coordinator compresses
// each frame, writes the compressed bytes to the corresponding per-shard
// nodes_{NNN}.jsonl.zst BufWriter, updates per-shard compressed offsets, hashes
// per-shard bytes for per-file integrity, and emits a single shared index line
// into nodes.idx.jsonl containing the shard and byte_offset for that frame.
//
// This single-coordinator design preserves deterministic ordering of index
// emission and keeps implementation simple and crash-safe.
#[derive(Debug)]
struct ShardedWriterOutcome {
    total_lines: u64,
    frame_count: u64,
    nodes_sha256_list: Vec<String>, // per-shard digests (hex), in shard order
    index_sha256_hex: Option<String>,
}

pub struct AsyncShardedZstdFramesJsonlWriter {
    // producer-side frame builder (same shape as AsyncZstdFramesJsonlWriter)
    frame_buf: Vec<u8>,
    frame_lines_target: usize,
    frame_max_bytes: usize,
    frame_lines_count: usize,
    cur_turn_min: Option<u8>,
    cur_turn_max: Option<u8>,
    next_frame_id: u64,
    global_line_cursor: u64,
 
    // channel to coordinator
    tx: Sender<FrameMsg>,
 
    // join handle for the coordinator
    writer_join: Option<thread::JoinHandle<io::Result<ShardedWriterOutcome>>>,
 
    // shard count and final fsync policy (retained for coordinator use)
    shards: usize,
    sync_final: bool,
    hashing: bool,
}

impl AsyncShardedZstdFramesJsonlWriter {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        shard_files: Vec<File>, // length == shards
        idx_file: Option<File>, // shared index file (nodes.idx.jsonl)
        buf_bytes: usize,
        zstd_level: i32,
        zstd_threads: usize,
        frame_lines_target: usize,
        frame_max_bytes: usize,
        writer_queue_frames: usize,
        sync_final: bool,
        hashing: bool,
    ) -> Self {
        let shards = shard_files.len().max(1);
        let (tx, rx) = bounded::<FrameMsg>(writer_queue_frames.max(1));

        // Spawn coordinator thread
        let join = thread::spawn(move || -> io::Result<ShardedWriterOutcome> {
            // Prepare per-shard writers, hashers and pos counters
            let mut nodes_writers: Vec<BufWriter<File>> = shard_files
                .into_iter()
                .map(|f| BufWriter::with_capacity(buf_bytes.max(8 * 1024 * 1024), f))
                .collect();

            let mut hasher_nodes: Vec<Sha256> = vec![Sha256::new(); shards];
            let mut compressed_pos: Vec<u64> = vec![0u64; shards];

            // Shared index writer + hasher
            let mut idx_writer = idx_file.map(|f| BufWriter::with_capacity(buf_bytes.max(1024 * 1024), f));
            let mut hasher_idx = idx_writer.as_ref().map(|_| Sha256::new());

            let mut total_lines: u64 = 0;
            let mut frame_count: u64 = 0;

            while let Ok(msg) = rx.recv() {
                match msg {
                    FrameMsg::Data(f) => {
                        let shard = (f.id as usize) % shards;
                        let offset = compressed_pos[shard];
 
                        // Compress frame into corresponding shard writer while hashing and counting
                        {
                            let mut counting_writer = HashingFileWrite {
                                inner: &mut nodes_writers[shard],
                                hasher: &mut hasher_nodes[shard],
                                pos: &mut compressed_pos[shard],
                                hashing,
                            };
                            let mut encoder = zstd::stream::write::Encoder::new(&mut counting_writer, zstd_level)?;
                            let _ = encoder.multithread(zstd_threads.max(1) as u32);
                            encoder.write_all(&f.uncompressed)?;
                            let _ = encoder.finish();
                        }
 
                        // Emit one index line to shared index with shard field and global frame id
                        if let (Some(idxw), Some(h)) = (idx_writer.as_mut(), hasher_idx.as_mut()) {
                            let line = serde_json::json!({
                                "frame": f.id,
                                "shard": shard,
                                "line_start": f.line_start,
                                "byte_offset": offset,
                                "lines": f.line_count as u64,
                                "turn_min": f.turn_min,
                                "turn_max": f.turn_max
                            });
                            let mut tmp = serde_json::to_vec(&line).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
                            tmp.push(b'\n');
                            idxw.write_all(&tmp)?;
                            if hashing {
                                h.update(&tmp);
                            }
                        }
 
                        frame_count = frame_count.saturating_add(1);
                        total_lines = total_lines.saturating_add(f.line_count as u64);
                    }
                    FrameMsg::End => break,
                }
            }

            // Final flushes
            for w in nodes_writers.iter_mut() {
                w.flush()?;
            }
            if let Some(iw) = idx_writer.as_mut() {
                iw.flush()?;
            }

            if sync_final {
                // fsync all files
                for w in nodes_writers.iter() {
                    w.get_ref().sync_all()?;
                }
                if let Some(iw) = idx_writer.as_ref() {
                    iw.get_ref().sync_all()?;
                }
            }

            // Finalize per-shard digests
            let nodes_sha256_list: Vec<String> = if hashing {
                hasher_nodes
                    .into_iter()
                    .map(|h| hex::encode(h.finalize()))
                    .collect()
            } else {
                Vec::new()
            };

            let index_sha = hasher_idx.map(|h| hex::encode(h.finalize()));

            Ok(ShardedWriterOutcome {
                total_lines,
                frame_count,
                nodes_sha256_list,
                index_sha256_hex: index_sha,
            })
        });

        Self {
            frame_buf: Vec::with_capacity(frame_max_bytes.max(128 * 1024 * 1024)),
            frame_lines_target: frame_lines_target.max(1),
            frame_max_bytes: frame_max_bytes.max(1 * 1024 * 1024),
            frame_lines_count: 0,
            cur_turn_min: None,
            cur_turn_max: None,
            next_frame_id: 0,
            global_line_cursor: 0,
            tx,
            writer_join: Some(join),
            shards,
            sync_final,
            hashing,
        }
    }

    fn emit_frame(&mut self) -> io::Result<()> {
        if self.frame_lines_count == 0 {
            return Ok(());
        }
        let turn_min = self.cur_turn_min.unwrap_or(0);
        let turn_max = self.cur_turn_max.unwrap_or(turn_min);
        let line_start = self.global_line_cursor;
        let line_count_u32 = self.frame_lines_count as u32;

        let buf = std::mem::replace(
            &mut self.frame_buf,
            Vec::with_capacity(self.frame_max_bytes.max(128 * 1024 * 1024)),
        );

        let frame = Frame {
            id: self.next_frame_id,
            uncompressed: buf,
            turn_min,
            turn_max,
            line_start,
            line_count: line_count_u32,
        };

        self.next_frame_id = self.next_frame_id.saturating_add(1);
        self.global_line_cursor = self.global_line_cursor.saturating_add(self.frame_lines_count as u64);
        self.frame_lines_count = 0;
        self.cur_turn_min = None;
        self.cur_turn_max = None;

        self.tx
            .send(FrameMsg::Data(frame))
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "sharded writer channel closed"))
    }
}

impl GraphJsonlSink for AsyncShardedZstdFramesJsonlWriter {
    fn write_line(&mut self, json_line: &[u8], turn: u8) -> io::Result<()> {
        self.frame_buf.extend_from_slice(json_line);
        self.frame_buf.push(b'\n');

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

        if self.frame_lines_count >= self.frame_lines_target || self.frame_buf.len() >= self.frame_max_bytes {
            self.emit_frame()?;
        }
        Ok(())
    }

    fn finish_mut(&mut self) -> io::Result<GraphSinkStats> {
        // Flush any pending frame
        self.emit_frame()?;

        // signal end
        self.tx
            .send(FrameMsg::End)
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "sharded writer channel closed"))?;

        // join coordinator
        let handle = self.writer_join.take().expect("sharded writer join present");
        let outcome = handle.join().map_err(|_| io::Error::new(io::ErrorKind::Other, "sharded writer thread panicked"))??;

        Ok(GraphSinkStats {
            total_lines: outcome.total_lines,
            frame_count: outcome.frame_count,
            nodes_sha256_hex: if self.hashing { outcome.nodes_sha256_list.get(0).cloned() } else { None },
            index_sha256_hex: outcome.index_sha256_hex,
            nodes_sha256_list: if self.hashing { Some(outcome.nodes_sha256_list) } else { None },
        })
    }
}