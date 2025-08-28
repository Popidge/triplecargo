use crc32fast::Hasher as Crc32;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::persist::{DbHeader, SolvedEntry, FORMAT_VERSION, save_db, load_db};

const STREAM_MAGIC: [u8; 8] = *b"TCDBSTRM";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StreamCompression {
    None,
    Lz4,
    Zstd,
}

impl StreamCompression {
    #[inline]
    fn tag(self) -> u8 {
        match self {
            StreamCompression::None => 0,
            StreamCompression::Lz4 => 1,
            StreamCompression::Zstd => 2,
        }
    }

    #[inline]
    fn from_tag(tag: u8) -> Result<Self, String> {
        match tag {
            0 => Ok(StreamCompression::None),
            1 => Ok(StreamCompression::Lz4),
            2 => Ok(StreamCompression::Zstd),
            _ => Err(format!("Unknown compression tag: {tag}")),
        }
    }
}

#[derive(Debug)]
pub struct StreamWriter {
    file: File,
    compression: StreamCompression,
    batch_size: usize,
    buf: Vec<(u128, SolvedEntry)>,
    next_seq: u64,
    header: DbHeader,
}

impl StreamWriter {
    pub fn create<P: AsRef<Path>>(
        path: P,
        header: &DbHeader,
        compression: StreamCompression,
        batch_size: usize,
    ) -> Result<Self, String> {
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .read(true)
            .open(path.as_ref())
            .map_err(|e| format!("open stream for write error: {e}"))?;

        write_stream_header(&mut file, header, compression)?;

        Ok(Self {
            file,
            compression,
            batch_size: batch_size.max(1),
            buf: Vec::with_capacity(batch_size.min(1024)),
            next_seq: 0,
            header: header.clone(),
        })
    }

    #[inline]
    pub fn header(&self) -> &DbHeader {
        &self.header
    }

    #[inline]
    pub fn compression(&self) -> StreamCompression {
        self.compression
    }

    #[inline]
    pub fn push(&mut self, key: u128, entry: SolvedEntry) -> Result<(), String> {
        self.buf.push((key, entry));
        if self.buf.len() >= self.batch_size {
            self.flush_batch()?;
        }
        Ok(())
    }

    pub fn flush_batch(&mut self) -> Result<(), String> {
        if self.buf.is_empty() {
            return Ok(());
        }

        // Deterministic order: sort by key ascending
        self.buf.sort_by_key(|(k, _)| *k);

        // Serialize uncompressed payload = bincode(Vec<(u128, SolvedEntry)>)
        let payload = bincode::serialize(&self.buf)
            .map_err(|e| format!("bincode serialize frame error: {e}"))?;

        // CRC32 on uncompressed payload
        let mut hasher = Crc32::new();
        hasher.update(&payload);
        let crc = hasher.finalize();

        // Capture uncompressed length before moving payload
        let ulen = payload.len() as u64;

        // Optional compression
        let (comp_tag, body_bytes) = match self.compression {
            StreamCompression::None => (self.compression.tag(), payload),
            StreamCompression::Lz4 => {
                let compressed = lz4_flex::block::compress(&payload);
                (self.compression.tag(), compressed)
            }
            StreamCompression::Zstd => {
                // Fixed level for determinism; single-threaded by default
                let compressed =
                    zstd::encode_all(std::io::Cursor::new(&payload), 3).map_err(|e| format!("zstd encode error: {e}"))?;
                (self.compression.tag(), compressed)
            }
        };

        let clen = body_bytes.len() as u64;

        // Frame header: seq:u64, comp:u8, ulen:u64, clen:u64, crc:u32
        self.file
            .write_all(&self.next_seq.to_le_bytes())
            .map_err(|e| format!("write frame seq error: {e}"))?;
        self.file
            .write_all(&[comp_tag])
            .map_err(|e| format!("write frame comp tag error: {e}"))?;
        self.file
            .write_all(&ulen.to_le_bytes())
            .map_err(|e| format!("write frame ulen error: {e}"))?;
        self.file
            .write_all(&clen.to_le_bytes())
            .map_err(|e| format!("write frame clen error: {e}"))?;
        self.file
            .write_all(&crc.to_le_bytes())
            .map_err(|e| format!("write frame crc error: {e}"))?;

        // Body
        self.file
            .write_all(&body_bytes)
            .map_err(|e| format!("write frame body error: {e}"))?;

        self.next_seq = self.next_seq.saturating_add(1);
        self.buf.clear();

        Ok(())
    }

    pub fn flush_all(&mut self) -> Result<(), String> {
        self.flush_batch()?;
        self.file.flush().map_err(|e| format!("file flush error: {e}"))?;
        Ok(())
    }

    pub fn sync_all(&mut self) -> Result<(), String> {
        self.flush_batch()?;
        self.file.sync_all().map_err(|e| format!("file sync_all error: {e}"))?;
        Ok(())
    }
}

fn write_stream_header(file: &mut File, header: &DbHeader, compression: StreamCompression) -> Result<(), String> {
    // Magic
    file.write_all(&STREAM_MAGIC)
        .map_err(|e| format!("write stream magic error: {e}"))?;
    // Version (FORMAT_VERSION)
    file.write_all(&FORMAT_VERSION.to_le_bytes())
        .map_err(|e| format!("write stream version error: {e}"))?;
    // Compression default tag
    file.write_all(&[compression.tag()])
        .map_err(|e| format!("write stream compression tag error: {e}"))?;
    // Header blob length (u32 LE) + bincode(DbHeader)
    let hdr_bytes =
        bincode::serialize(header).map_err(|e| format!("bincode serialize header error: {e}"))?;
    let hdr_len = hdr_bytes.len() as u32;
    file.write_all(&hdr_len.to_le_bytes())
        .map_err(|e| format!("write header length error: {e}"))?;
    file.write_all(&hdr_bytes)
        .map_err(|e| format!("write header blob error: {e}"))?;
    Ok(())
}

#[derive(Debug)]
pub struct StreamReader {
    file: File,
    compression_default: StreamCompression,
    pub header: DbHeader,
    next_seq_expected: u64,
    data_offset: u64,
    path: PathBuf,
}

impl StreamReader {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let mut file = OpenOptions::new()
            .read(true)
            .open(&path)
            .map_err(|e| format!("open stream for read error: {e}"))?;

        // Read header
        let mut magic = [0u8; 8];
        file.read_exact(&mut magic)
            .map_err(|e| format!("read magic error: {e}"))?;
        if magic != STREAM_MAGIC {
            return Err("invalid stream magic".into());
        }

        let mut ver = [0u8; 4];
        file.read_exact(&mut ver)
            .map_err(|e| format!("read version error: {e}"))?;
        let _version = u32::from_le_bytes(ver);
        if _version != FORMAT_VERSION {
            return Err(format!(
                "stream version mismatch: got {}, expected {}",
                _version, FORMAT_VERSION
            ));
        }

        let mut comp = [0u8; 1];
        file.read_exact(&mut comp)
            .map_err(|e| format!("read compression tag error: {e}"))?;
        let compression_default = StreamCompression::from_tag(comp[0])?;

        let mut hlen = [0u8; 4];
        file.read_exact(&mut hlen)
            .map_err(|e| format!("read header length error: {e}"))?;
        let header_len = u32::from_le_bytes(hlen) as usize;

        let mut hdr_bytes = vec![0u8; header_len];
        file.read_exact(&mut hdr_bytes)
            .map_err(|e| format!("read header bytes error: {e}"))?;
        let header: DbHeader = bincode::deserialize(&hdr_bytes)
            .map_err(|e| format!("bincode header deserialize error: {e}"))?;

        let data_offset = file.seek(SeekFrom::Current(0)).map_err(|e| format!("seek error: {e}"))?;

        Ok(Self {
            file,
            compression_default,
            header,
            next_seq_expected: 0,
            data_offset,
            path: path.as_ref().to_path_buf(),
        })
    }

    /// Reads next frame. Returns Ok(None) at EOF.
    pub fn read_next_frame(&mut self) -> Result<Option<Vec<(u128, SolvedEntry)>>, String> {
        // Attempt to read seq (u64). If EOF, return None.
        let mut seq_bytes = [0u8; 8];
        match self.file.read_exact(&mut seq_bytes) {
            Ok(()) => {}
            Err(e) => {
                if e.kind() == std::io::ErrorKind::UnexpectedEof {
                    return Ok(None);
                } else {
                    return Err(format!("read seq error: {e}"));
                }
            }
        };
        let seq = u64::from_le_bytes(seq_bytes);
        if seq != self.next_seq_expected {
            return Err(format!(
                "sequence mismatch: got {}, expected {}",
                seq, self.next_seq_expected
            ));
        }

        let mut comp_tag = [0u8; 1];
        self.file
            .read_exact(&mut comp_tag)
            .map_err(|e| format!("read comp tag error: {e}"))?;
        let comp = StreamCompression::from_tag(comp_tag[0])?;

        let mut ulen = [0u8; 8];
        self.file
            .read_exact(&mut ulen)
            .map_err(|e| format!("read uncompressed len error: {e}"))?;
        let ulen = u64::from_le_bytes(ulen) as usize;

        let mut clen = [0u8; 8];
        self.file
            .read_exact(&mut clen)
            .map_err(|e| format!("read compressed len error: {e}"))?;
        let clen = u64::from_le_bytes(clen) as usize;

        let mut crc_bytes = [0u8; 4];
        self.file
            .read_exact(&mut crc_bytes)
            .map_err(|e| format!("read crc error: {e}"))?;
        let crc_expected = u32::from_le_bytes(crc_bytes);

        let mut body = vec![0u8; clen];
        self.file
            .read_exact(&mut body)
            .map_err(|e| format!("read body error: {e}"))?;

        // Decompress if needed
        let payload = match comp {
            StreamCompression::None => {
                if body.len() != ulen {
                    return Err(format!(
                        "length mismatch for uncompressed frame: body={}, ulen={}",
                        body.len(),
                        ulen
                    ));
                }
                body
            }
            StreamCompression::Lz4 => {
                // Deterministic decode with known size. Do NOT attempt size-prepended variant.
                lz4_flex::block::decompress(&body, ulen)
                    .map_err(|e| format!("lz4 decompress error: {e}"))?
            }
            StreamCompression::Zstd => {
                zstd::decode_all(std::io::Cursor::new(&body))
                    .map_err(|e| format!("zstd decode error: {e}"))?
            }
        };

        if payload.len() != ulen {
            return Err(format!(
                "payload length mismatch: got {}, expected {}",
                payload.len(),
                ulen
            ));
        }

        // CRC verify
        let mut hasher = Crc32::new();
        hasher.update(&payload);
        let crc = hasher.finalize();
        if crc != crc_expected {
            return Err("crc mismatch on frame payload".into());
        }

        let items: Vec<(u128, SolvedEntry)> =
            bincode::deserialize(&payload).map_err(|e| format!("bincode payload deserialize error: {e}"))?;

        self.next_seq_expected = self.next_seq_expected.saturating_add(1);

        Ok(Some(items))
    }

    /// Deterministically compacts the entire stream into a BTreeMap.
    pub fn read_all_compacted(&mut self) -> Result<BTreeMap<u128, SolvedEntry>, String> {
        // Reset to data start
        self.file
            .seek(SeekFrom::Start(self.data_offset))
            .map_err(|e| format!("seek to data offset error: {e}"))?;
        self.next_seq_expected = 0;

        let mut merged: BTreeMap<u128, SolvedEntry> = BTreeMap::new();

        while let Some(items) = self.read_next_frame()? {
            for (k, v) in items {
                match merged.get(&k) {
                    None => {
                        merged.insert(k, v);
                    }
                    Some(old) => {
                        // Deterministic preference for SolvedEntry vs SolvedEntry
                        if prefer_solved_new_over_old(&v, old) {
                            merged.insert(k, v);
                        }
                    }
                }
            }
        }

        Ok(merged)
    }
}

// Deterministic tie-breaking using only SolvedEntry fields
// - Higher depth wins
// - If equal depth, Some(best_move) > None
// - If both Some, prefer lexicographically smaller (cell, then card_id)
// - If still equal or both None, prefer smaller value to keep order stable
#[inline]
fn prefer_solved_new_over_old(new: &SolvedEntry, old: &SolvedEntry) -> bool {
    if new.depth > old.depth {
        return true;
    }
    if new.depth < old.depth {
        return false;
    }
    match (new.best_move, old.best_move) {
        (Some(a), Some(b)) => {
            if a.cell != b.cell {
                return a.cell < b.cell;
            }
            if a.card_id != b.card_id {
                return a.card_id < b.card_id;
            }
            new.value < old.value
        }
        (Some(_), None) => true,
        (None, Some(_)) => false,
        (None, None) => new.value < old.value,
    }
}

/// Convenience: compact a stream file into a BTreeMap.
pub fn compact_stream_to_map<P: AsRef<Path>>(path: P) -> Result<(DbHeader, BTreeMap<u128, SolvedEntry>), String> {
    let mut reader = StreamReader::open(&path)?;
    let header = reader.header.clone();
    let merged = reader.read_all_compacted()?;
    Ok((header, merged))
}

/// Compact a stream file directly into a legacy DB file using persist::save_db.
pub fn compact_stream_to_db_file<P1: AsRef<Path>, P2: AsRef<Path>>(stream_path: P1, out_path: P2) -> Result<(), String> {
    let (header, merged) = compact_stream_to_map(stream_path)?;
    save_db(out_path, &header, &merged)
}

/// Compact a stream file together with a baseline legacy DB file into a new legacy DB file.
/// Baseline entries are merged first; stream entries then override using deterministic preference.
pub fn compact_stream_with_baseline_to_db_file<P1: AsRef<Path>, P2: AsRef<Path>, P3: AsRef<Path>>(
    stream_path: P1,
    baseline_db_path: P2,
    out_path: P3,
) -> Result<(), String> {
    let (stream_header, stream_map) = compact_stream_to_map(&stream_path)?;
    let (base_header, base_map) = load_db(&baseline_db_path)
        .map_err(|e| format!("load baseline db error: {e}"))?;

    // Basic header compatibility checks for determinism
    if stream_header.version != base_header.version
        || stream_header.rules != base_header.rules
        || stream_header.elements_mode as u8 != base_header.elements_mode as u8
        || stream_header.seed != base_header.seed
        || stream_header.start_player as u8 != base_header.start_player as u8
        || stream_header.hands_a != base_header.hands_a
        || stream_header.hands_b != base_header.hands_b
        || stream_header.cards_fingerprint != base_header.cards_fingerprint
    {
        return Err("baseline checkpoint header incompatible with current stream header".into());
    }

    // Merge deterministically
    let mut merged = base_map;
    for (k, v_new) in stream_map {
        match merged.get(&k) {
            None => {
                merged.insert(k, v_new);
            }
            Some(old) => {
                if prefer_solved_new_over_old(&v_new, old) {
                    merged.insert(k, v_new);
                }
            }
        }
    }

    save_db(out_path, &stream_header, &merged)
}