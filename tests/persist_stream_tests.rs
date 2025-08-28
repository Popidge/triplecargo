use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;

use triplecargo::persist::{load_db, DbHeader, ElementsMode, SolvedEntry};
use triplecargo::persist_stream::{compact_stream_to_db_file, StreamCompression, StreamReader, StreamWriter};
use triplecargo::{Owner, Rules};

fn make_header() -> DbHeader {
    DbHeader {
        version: triplecargo::persist::FORMAT_VERSION,
        rules: Rules::default(),
        elements_mode: ElementsMode::None,
        seed: 0xABCD_EF01_2345_6789,
        start_player: Owner::A,
        hands_a: [1, 2, 3, 4, 5],
        hands_b: [6, 7, 8, 9, 10],
        cards_fingerprint: 0x1111_2222_3333_4444_5555_6666_7777_8888u128,
    }
}

fn sample_entries(n: usize) -> BTreeMap<u128, SolvedEntry> {
    let mut map = BTreeMap::new();
    for i in 0..n {
        let key = (i as u128) * 0x1_0000_0001u128 + 0xABC_u128;
        let entry = SolvedEntry {
            value: ((i as i32 % 7) - 3) as i8,
            best_move: Some(triplecargo::Move { card_id: (i % 10) as u16, cell: (i % 9) as u8 }),
            depth: (i % 9) as u8,
        };
        map.insert(key, entry);
    }
    map
}

fn tmp_file(name: &str, ext: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    p.push(format!("{name}-{ts}.{ext}"));
    p
}

#[test]
fn stream_roundtrip_none_lz4_zstd() -> Result<(), String> {
    for comp in [StreamCompression::None, StreamCompression::Lz4, StreamCompression::Zstd] {
        let header = make_header();
        let entries = sample_entries(10_000);

        let stream_path = tmp_file("tc-stream-rt", "stream");
        let out_db = tmp_file("tc-stream-rt", "bin");

        {
            let mut writer = StreamWriter::create(&stream_path, &header, comp, 1000)?;
            // deterministic order already guaranteed by map iteration
            for (k, v) in &entries {
                writer.push(*k, v.clone())?;
            }
            writer.flush_all()?;
        }

        // Reader compaction to map
        let mut reader = StreamReader::open(&stream_path)?;
        let got = reader.read_all_compacted()?;
        assert_eq!(entries.len(), got.len(), "map length differs for comp={:?}", comp);
        for (k, v) in &entries {
            let g = got.get(k).expect("missing key from compaction");
            assert_eq!(v.value, g.value);
            assert_eq!(v.best_move, g.best_move);
            assert_eq!(v.depth, g.depth);
        }

        // Compact directly to legacy DB and load back
        compact_stream_to_db_file(&stream_path, &out_db)?;
        let (hdr2, map2) = load_db(&out_db).map_err(|e| format!("load_db error: {e}"))?;
        // header identity
        assert_eq!(hdr2.version, header.version);
        assert_eq!(hdr2.rules, header.rules);
        assert_eq!(hdr2.elements_mode as u8, header.elements_mode as u8);
        assert_eq!(hdr2.seed, header.seed);
        assert_eq!(hdr2.start_player as u8, header.start_player as u8);
        assert_eq!(hdr2.hands_a, header.hands_a);
        assert_eq!(hdr2.hands_b, header.hands_b);
        assert_eq!(hdr2.cards_fingerprint, header.cards_fingerprint);

        assert_eq!(map2.len(), entries.len());
        for (k, v) in &entries {
            let g = map2.get(k).expect("missing key from legacy db");
            assert_eq!(v.value, g.value);
            assert_eq!(v.best_move, g.best_move);
            assert_eq!(v.depth, g.depth);
        }

        // Cleanup
        let _ = fs::remove_file(&stream_path);
        let _ = fs::remove_file(&out_db);
    }
    Ok(())
}

#[test]
fn stream_crc_detection() -> Result<(), String> {
    let header = make_header();
    let entries = sample_entries(2048);

    let stream_path = tmp_file("tc-stream-crc", "stream");

    {
        let mut writer = StreamWriter::create(&stream_path, &header, StreamCompression::None, 256)?;
        for (k, v) in &entries {
            writer.push(*k, v.clone())?;
        }
        writer.flush_all()?;
    }

    // Corrupt one byte near the end of the file to trigger CRC mismatch
    {
        let mut bytes = Vec::new();
        {
            let mut f = OpenOptions::new().read(true).open(&stream_path).map_err(|e| e.to_string())?;
            f.read_to_end(&mut bytes).map_err(|e| e.to_string())?;
        }
        if let Some(last) = bytes.last_mut() {
            *last ^= 0xFF;
        }
        let mut f = OpenOptions::new().write(true).truncate(true).open(&stream_path).map_err(|e| e.to_string())?;
        f.write_all(&bytes).map_err(|e| e.to_string())?;
        f.flush().map_err(|e| e.to_string())?;
    }

    let mut reader = StreamReader::open(&stream_path)?;
    let res = reader.read_all_compacted();
    assert!(res.is_err(), "expected CRC mismatch error after corruption");

    let _ = fs::remove_file(&stream_path);
    Ok(())
}