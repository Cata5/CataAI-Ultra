| Feature | Frontend | API | Backend | Runtime | Persistence | Overall |
|---|---|---|---|---|---|---|
| Start training — RoCanon decoder | PASS | PASS | PASS | PASS | PASS | **PASS** |
| Start training — custom decoder/hybrid network | PASS | PASS | PASS | NOT TESTED | N/A | **NOT TESTED** |
| Start training — MLP / CNN / classifier networks | PASS | PASS | FAIL | N/A | N/A | **FAIL** |
| Pause training | PASS | PASS | PASS | NOT TESTED | N/A | **NOT TESTED** |
| Resume training | PASS | PASS | PASS | NOT TESTED | N/A | **NOT TESTED** |
| Stop training | PASS | PASS | PASS | NOT TESTED | N/A | **NOT TESTED** |
| Manual checkpoint button | PASS | PASS | PASS | NOT TESTED | PASS | **NOT TESTED** |
| Automatic / best / periodic checkpoints | N/A | N/A | PASS | PASS | PASS | **PASS** |
| Checkpoint listing | PASS | PASS | PASS | PASS | N/A | **PASS** |
| Checkpoint weight loading | PARTIAL | PASS | PASS | NOT TESTED | N/A | **PARTIAL** |
| Optimizer / scheduler / RNG restoration | N/A | N/A | FAIL | NOT TESTED | N/A | **FAIL** |
| Delete checkpoint | PASS | PASS | PASS | PASS | PASS | **PASS** |
| Dataset upload + validation + analysis | PASS | PASS | PASS | PASS | PASS | **PASS** |
| Dataset list / stats API | PASS | PASS | PASS | PASS | PASS | **PASS** |
| Dataset preview in UI | NOT IMPLEMENTED | PASS | PASS | N/A | PASS | **PARTIAL** |
| Select an uploaded dataset for training | FAIL | PASS | FAIL | N/A | PASS | **FAIL** |
| Sample a dataset in training | N/A | PASS | FAIL | N/A | PASS | **FAIL** |
| Tokenizer inspector | PASS | PASS | PASS | PASS | N/A | **PASS** |
| Tokenizer corpus / vocab | N/A | N/A | PASS | PASS | N/A | **PASS** |
| Live text generation | PASS | PASS | PASS | BLOCKED | N/A | **PARTIAL** |
| Evaluation model / checkpoint selection | NOT IMPLEMENTED | FAIL | PARTIAL | N/A | N/A | **FAIL** |
| Side-by-side checkpoint comparison | PASS | PASS | PASS | BLOCKED | N/A | **PARTIAL** |
| Benchmark suite execution | PASS | PASS | PASS | BLOCKED | PASS | **PARTIAL** |
| Benchmark history retrieval | NOT IMPLEMENTED | PASS | PASS | N/A | PASS | **FAIL** |
| NVML hardware telemetry | PASS | PASS | PASS | PASS | N/A | **PASS** |
| PyTorch allocator statistics | PASS | PASS | PASS | PASS | N/A | **PASS** |
| VRAM alert evaluation | PASS | PASS | PASS | PASS | N/A | **PASS** |
| Alert delivery over WebSocket | PASS | PASS | PASS | PASS | PASS | **PASS** |
| Alert history from DB | NOT IMPLEMENTED | PASS | PASS | N/A | PASS | **PARTIAL** |
| Architecture templates instantiate + forward | PASS | PASS | PASS | PASS | N/A | **PASS** |
| CNN template forward pass | PASS | PASS | FAIL | N/A | N/A | **FAIL** |
| Add / remove / reorder / connect components | NOT IMPLEMENTED | N/A | N/A | N/A | N/A | **NOT IMPLEMENTED** |
| Visual computation graph | PASS | N/A | PASS | N/A | N/A | **PASS** |
| Save architecture to registry | PASS | PASS | PARTIAL | PASS | PASS | **PARTIAL** |
| Load / clone / delete architecture | PASS | PASS | PASS | PASS | PASS | **PASS** |
| Export / import architecture | NOT IMPLEMENTED | NOT IMPLEMENTED | NOT IMPLEMENTED | N/A | N/A | **NOT IMPLEMENTED** |
| Architecture versioning | NOT IMPLEMENTED | N/A | NOT IMPLEMENTED | N/A | PARTIAL | **NOT IMPLEMENTED** |
| Parameter count (total) | PASS | PASS | PASS | PASS | N/A | **PASS** |
| Parameter breakdown per component | PASS | PASS | PARTIAL | N/A | N/A | **PARTIAL** |
| VRAM estimator | PASS | PASS | PARTIAL | PASS | N/A | **PARTIAL** |
| Architecture preflight (forward/backward) | PASS | PASS | PASS | PASS | N/A | **PASS** |
| Custom module discovery | PASS | PASS | PASS | PASS | N/A | **PARTIAL** |
| Use a custom module in a model | NOT IMPLEMENTED | NOT IMPLEMENTED | NOT IMPLEMENTED | N/A | N/A | **NOT IMPLEMENTED** |
| Generic ModelFactory build/forward/backward | N/A | N/A | PASS | PASS | N/A | **PASS** |
| Create / list pipelines | NOT IMPLEMENTED | PASS | PASS | PASS | PASS | **PARTIAL** |
| Multi-stage pipeline execution | NOT IMPLEMENTED | NOT IMPLEMENTED | NOT IMPLEMENTED | N/A | NOT IMPLEMENTED | **NOT IMPLEMENTED** |
| Scheduler curve visualizer | PASS | PASS | PASS | PASS | N/A | **PASS** |
| Runs history table | PASS | PASS | PASS | PASS | PASS | **PASS** |
| Live metric charts | PASS | PASS | PASS | PASS | PARTIAL | **PARTIAL** |
| Settings persistence | FAIL | PASS | PASS | N/A | PARTIAL | **PARTIAL** |
| Live learning-rate adjustment | PASS | PASS | PASS | NOT TESTED | N/A | **NOT TESTED** |
| WebSocket event delivery | PASS | PASS | PASS | PASS | N/A | **PASS** |
| State persistence across restart | N/A | N/A | PASS | PASS | PASS | **PASS** |
| Security: checkpoint deserialization | N/A | N/A | FAIL | N/A | N/A | **FAIL** |
| Security: path traversal in network ids | N/A | FAIL | PASS | PASS | N/A | **FAIL** |
