//! Encryption-at-rest for desktop-persisted credentials.
//!
//! The key file and ciphertext format intentionally match
//! `src/security/secrets.py`, so the Python control plane and the Tauri shell
//! can read the same database. Windows protects the AES key with per-user
//! DPAPI; other platforms keep a 0600 key file next to the database.

use aes_gcm::aead::{Aead, KeyInit, OsRng, Payload, rand_core::RngCore};
use aes_gcm::{Aes256Gcm, Nonce};
use base64::engine::general_purpose::STANDARD;
use base64::Engine as _;
use serde_json::json;
use std::fs;
use std::path::{Path, PathBuf};

pub const PREFIX: &str = "enc:v1:";
pub const REDACTED: &str = "__LLMMAKER_SECRET_SET__";
pub const SECRET_KEYS: [&str; 2] = ["discord_webhook_url", "telegram_bot_token"];
const AAD: &[u8] = b"llmmaker-settings-v1";

pub fn is_secret_key(key: &str) -> bool {
    SECRET_KEYS.contains(&key)
}

pub struct SecretStore {
    key: [u8; 32],
    key_path: PathBuf,
}

impl SecretStore {
    pub fn load_or_create(key_path: &Path) -> Result<Self, String> {
        let key = if key_path.exists() {
            Self::load_key(key_path)?
        } else {
            Self::create_key(key_path)?
        };
        Ok(Self {
            key,
            key_path: key_path.to_path_buf(),
        })
    }

    pub fn key_path(&self) -> &Path {
        &self.key_path
    }

    fn load_key(path: &Path) -> Result<[u8; 32], String> {
        let text = fs::read_to_string(path)
            .map_err(|e| format!("cannot read key file {}: {}", path.display(), e))?;
        let record: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| format!("cannot parse key file {}: {}", path.display(), e))?;
        if record.get("version").and_then(|v| v.as_i64()) != Some(1) {
            return Err(format!("unsupported key file version in {}", path.display()));
        }
        let encoded = record
            .get("key")
            .and_then(|v| v.as_str())
            .ok_or_else(|| format!("key file {} has no key", path.display()))?;
        let stored = STANDARD
            .decode(encoded)
            .map_err(|e| format!("key file {} is not valid base64: {}", path.display(), e))?;
        let raw = match record.get("protection").and_then(|v| v.as_str()) {
            Some("dpapi") => dpapi(&stored, false)?,
            Some("raw") => stored,
            other => {
                return Err(format!(
                    "unsupported key protection {:?} in {}",
                    other,
                    path.display()
                ))
            }
        };
        raw.try_into().map_err(|_| {
            format!(
                "key file {} must contain exactly 32 bytes; restore the matching key",
                path.display()
            )
        })
    }

    fn create_key(path: &Path) -> Result<[u8; 32], String> {
        let mut key = [0u8; 32];
        OsRng.fill_bytes(&mut key);

        let (protection, stored) = if cfg!(windows) {
            ("dpapi", dpapi(&key, true)?)
        } else {
            ("raw", key.to_vec())
        };
        let record = json!({
            "version": 1,
            "protection": protection,
            "key": STANDARD.encode(stored),
        });
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("cannot create key directory {}: {}", parent.display(), e))?;
        }
        let tmp = path.with_extension("key.tmp");
        fs::write(&tmp, serde_json::to_vec(&record).map_err(|e| e.to_string())?)
            .map_err(|e| format!("cannot write key file {}: {}", tmp.display(), e))?;
        restrict_key_permissions(&tmp)?;
        fs::rename(&tmp, path)
            .map_err(|e| format!("cannot install key file {}: {}", path.display(), e))?;
        restrict_key_permissions(path)?;
        Ok(key)
    }

    pub fn is_encrypted(value: &str) -> bool {
        value.starts_with(PREFIX)
    }

    pub fn encrypt(&self, plaintext: &str) -> Result<String, String> {
        if Self::is_encrypted(plaintext) {
            return Ok(plaintext.to_string());
        }
        let cipher = Aes256Gcm::new_from_slice(&self.key).map_err(|e| e.to_string())?;
        let mut nonce = [0u8; 12];
        OsRng.fill_bytes(&mut nonce);
        let sealed = cipher
            .encrypt(
                Nonce::from_slice(&nonce),
                Payload {
                    msg: plaintext.as_bytes(),
                    aad: AAD,
                },
            )
            .map_err(|_| "cannot encrypt credential".to_string())?;
        let mut payload = Vec::with_capacity(nonce.len() + sealed.len());
        payload.extend_from_slice(&nonce);
        payload.extend_from_slice(&sealed);
        Ok(format!("{}{}", PREFIX, STANDARD.encode(payload)))
    }

    pub fn decrypt(&self, value: &str) -> Result<String, String> {
        let Some(encoded) = value.strip_prefix(PREFIX) else {
            return Ok(value.to_string());
        };
        let payload = STANDARD
            .decode(encoded)
            .map_err(|_| "stored secret is not valid base64".to_string())?;
        if payload.len() < 13 {
            return Err("stored secret is truncated".to_string());
        }
        let cipher = Aes256Gcm::new_from_slice(&self.key).map_err(|e| e.to_string())?;
        let plaintext = cipher
            .decrypt(
                Nonce::from_slice(&payload[..12]),
                Payload {
                    msg: &payload[12..],
                    aad: AAD,
                },
            )
            .map_err(|_| "stored secret failed authentication".to_string())?;
        String::from_utf8(plaintext).map_err(|_| "stored secret is not UTF-8".to_string())
    }
}

fn restrict_key_permissions(path: &Path) -> Result<(), String> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o600))
            .map_err(|e| format!("cannot secure {}: {}", path.display(), e))?;
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
    Ok(())
}

#[cfg(windows)]
fn dpapi(data: &[u8], protect: bool) -> Result<Vec<u8>, String> {
    use windows_sys::Win32::Foundation::LocalFree;
    use windows_sys::Win32::Security::Cryptography::{
        CRYPTPROTECT_UI_FORBIDDEN, CRYPT_INTEGER_BLOB, CryptProtectData, CryptUnprotectData,
    };

    let mut input = data.to_vec();
    let blob_in = CRYPT_INTEGER_BLOB {
        cbData: u32::try_from(input.len()).map_err(|_| "credential is too large".to_string())?,
        pbData: input.as_mut_ptr(),
    };
    let mut blob_out = CRYPT_INTEGER_BLOB::default();
    let ok = unsafe {
        if protect {
            CryptProtectData(
                &blob_in,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                CRYPTPROTECT_UI_FORBIDDEN,
                &mut blob_out,
            )
        } else {
            CryptUnprotectData(
                &blob_in,
                std::ptr::null_mut(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                CRYPTPROTECT_UI_FORBIDDEN,
                &mut blob_out,
            )
        }
    };
    if ok == 0 {
        return Err(format!(
            "DPAPI call failed (error {})",
            std::io::Error::last_os_error()
        ));
    }
    let result = unsafe {
        std::slice::from_raw_parts(blob_out.pbData, blob_out.cbData as usize).to_vec()
    };
    unsafe {
        LocalFree(blob_out.pbData as *mut core::ffi::c_void);
    }
    Ok(result)
}

#[cfg(not(windows))]
fn dpapi(_data: &[u8], _protect: bool) -> Result<Vec<u8>, String> {
    Err("DPAPI is only available on Windows".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_and_prefix() {
        let dir = std::env::temp_dir().join(format!("llmmaker-secret-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join(".secrets.key");
        let store = SecretStore::load_or_create(&path).unwrap();
        let sealed = store.encrypt("https://example.invalid/hook").unwrap();
        assert!(sealed.starts_with(PREFIX));
        assert_eq!(store.decrypt(&sealed).unwrap(), "https://example.invalid/hook");
        let reopened = SecretStore::load_or_create(&path).unwrap();
        assert_eq!(reopened.decrypt(&sealed).unwrap(), "https://example.invalid/hook");
        let _ = fs::remove_dir_all(&dir);
    }
}
