use std::path::Path;

use cwc_core::config::CwcConfig;
use cwc_core::error::Result;

/// Initialize a new CWC project: create cwc.toml and data directories.
pub fn init_project(
    project_dir: &Path,
    name: Option<&str>,
    endpoint: Option<&str>,
) -> Result<()> {
    let mut config = CwcConfig::default();

    if let Some(ep) = endpoint {
        config.model.llm_endpoint = ep.to_string();
    }

    // Create directories
    let data_dir = project_dir.join(&config.paths.data_dir);
    let index_dir = project_dir.join(&config.paths.index_dir);
    let models_dir = project_dir.join(&config.paths.models_dir);

    std::fs::create_dir_all(&data_dir)?;
    std::fs::create_dir_all(&index_dir)?;
    std::fs::create_dir_all(&models_dir)?;

    // Write config
    let config_path = project_dir.join("cwc.toml");
    let toml_str = toml::to_string_pretty(&config)
        .map_err(|e| cwc_core::error::CwcError::Config(format!("failed to serialize config: {e}")))?;

    let header = if let Some(n) = name {
        format!("# CWC Project: {n}\n\n")
    } else {
        String::from("# CWC Configuration\n\n")
    };

    std::fs::write(&config_path, format!("{header}{toml_str}"))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_init_creates_config_and_dirs() {
        let dir = std::env::temp_dir().join("cwc_test_init");
        let _ = fs::remove_dir_all(&dir);

        init_project(&dir, Some("test_project"), None).unwrap();

        // Config file exists
        let config_path = dir.join("cwc.toml");
        assert!(config_path.exists());

        // Directories exist
        assert!(dir.join("data").is_dir());
        assert!(dir.join("index").is_dir());
        assert!(dir.join("models").is_dir());

        // Config is valid TOML
        let contents = fs::read_to_string(&config_path).unwrap();
        assert!(contents.contains("CWC Project: test_project"));
        let _config: CwcConfig = toml::from_str(
            contents.lines().filter(|l| !l.starts_with('#')).collect::<Vec<_>>().join("\n").as_str()
        ).unwrap();

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_init_custom_endpoint() {
        let dir = std::env::temp_dir().join("cwc_test_init_ep");
        let _ = fs::remove_dir_all(&dir);

        init_project(&dir, None, Some("http://gpu:9090")).unwrap();

        let contents = fs::read_to_string(dir.join("cwc.toml")).unwrap();
        assert!(contents.contains("gpu:9090"));

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_init_idempotent() {
        let dir = std::env::temp_dir().join("cwc_test_init_idem");
        let _ = fs::remove_dir_all(&dir);

        init_project(&dir, None, None).unwrap();
        // Running again should not fail
        init_project(&dir, None, None).unwrap();

        assert!(dir.join("cwc.toml").exists());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_init_no_name_no_endpoint() {
        let dir = std::env::temp_dir().join("cwc_test_init_defaults");
        let _ = fs::remove_dir_all(&dir);

        init_project(&dir, None, None).unwrap();

        let contents = fs::read_to_string(dir.join("cwc.toml")).unwrap();
        assert!(contents.contains("CWC Configuration"));
        assert!(contents.contains("localhost:8080"));

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_init_config_is_parseable() {
        let dir = std::env::temp_dir().join("cwc_test_init_parseable");
        let _ = fs::remove_dir_all(&dir);

        init_project(&dir, Some("my project"), Some("http://gpu:1234")).unwrap();

        let contents = fs::read_to_string(dir.join("cwc.toml")).unwrap();
        // Strip comment lines and parse
        let clean: String = contents
            .lines()
            .filter(|l| !l.starts_with('#'))
            .collect::<Vec<_>>()
            .join("\n");
        let config: CwcConfig = toml::from_str(&clean).unwrap();
        assert_eq!(config.model.llm_endpoint, "http://gpu:1234");

        fs::remove_dir_all(&dir).unwrap();
    }
}
