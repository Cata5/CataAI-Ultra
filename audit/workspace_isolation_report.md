# Workspace Isolation & Resource Segregation Report

**Auditor:** ML Infrastructure Architecture Team  
**Scope:** File Storage, Database Integrity, Checkpoint Management, Multi-Project Coexistence  

---

## 1. Workspace Abstraction Architecture

In LLM Maker, each workspace acts as a self-contained environment for models, checkpoints, datasets, and experiment runs.

### Directory Layout

```
workspaces/
├── default/                        <-- Preserves repository root data
│   ├── checkpoints/
│   ├── data/
│   └── networks/registry/
├── workspace_alpha/
│   ├── manifest.json               <-- Workspace metadata, config, active model
│   ├── checkpoints/                <-- Isolated checkpoint weights (.pt, .json)
│   ├── datasets/                   <-- Uploaded datasets, splits, and token caches
│   ├── models/                     <-- Custom architectures and modular graphs
│   └── runs/                       <-- Run logs, metric artifacts, and exports
└── workspace_beta/
    ├── manifest.json
    ├── checkpoints/
    ├── datasets/
    ├── models/
    └── runs/
```

---

## 2. Relocation and Path Validation Safety

1. **Path Traversal Protection:** All workspace IDs are sanitized with regex `[^a-zA-Z0-9_\-]`, preventing directory traversal attacks (`../`).
2. **Backward Compatibility Mapping:** When `workspace_id == "default"`, `WorkspaceManager.get_workspace_paths()` maps directly to the root directories (`checkpoints/`, `data/`, `networks/registry/`), ensuring all historical runs, pre-existing checkpoints, and tokenized datasets remain 100% accessible.
3. **Safe Deletion Policy:** The `default` workspace cannot be deleted (`delete_workspace("default")` returns `False`). Deleting a custom workspace requires explicit confirmation and switches the active workspace back to `default` first.
4. **Export / Import Bundles:** Workspaces can be exported into portable JSON manifests containing all metadata, dataset references, and checkpoint indices, and restored into any LLM Maker installation.
