# Task 01 — Setup Virtual Environment

## Task Info

| Item | Detail |
|------|--------|
| **Sprint** | 0 |
| **Task ID** | S0-T01 |
| **Priority** | High |
| **Story Points** | 1 |
| **Status** | Belum Mulai |
| **Assignee** | - |

## Description
Menggunakan Python virtual environment yang DIBERIKAN oleh user di `c:\Users\Asus\OneDrive\Documents\GitHub\Project\.venv` untuk menghemat storage dan mempercepat setup. Pastikan kita activate tiap kali bekerja.

## Acceptance Criteria
- [ ] Tidak perlu membuat `venv` folder di dalam `skin_cancer_v3`
- [ ] Virtual environment global berhasil diaktivasi
- [ ] pip terupdate ke versi terbaru

## Implementation Steps
1. Buka terminal di folder project `skin_cancer_v3/`
2. Jalankan perintah aktivasi dari path absolute-nya:
   ```bash
   c:\Users\Asus\OneDrive\Documents\GitHub\Project\.venv\Scripts\activate
   ```
3. Update pip:
   ```bash
   python -m pip install --upgrade pip
   ```
4. Verifikasi:
   ```bash
   python --version
   pip --version
   ```

## Estimated Time
~15 menit

## Dependencies
- Python 3.9+ sudah terinstall di system

## Notes
- Kita memakai shared virtual environment (global project level).
- Pastikan tidak terhapus tidak sengaja sebab digunakan banyak project.
