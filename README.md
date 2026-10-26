# MindSeek Memory — AML Textual / Academic

这是独立的 Add/Search 服务部署包，运行时不依赖 Mem0 主仓库、评测脚本或数据集。

包内包括 Python memory 服务、部署入口、Dockerfile、Compose、HTTPS 代理、配置示例、方法说明、提交材料模板与文件校验清单。没有 `.env`、可用密钥、历史数据库、日志、测试集、答案或评分程序。

## 1. 准备部署配置

复制 `.env.example` 为 `.env`，仅在自己的部署服务器填写：

```powershell
Copy-Item .env.example .env
```

- `CHALLENGE_API_KEY`：自行生成的强随机服务访问密钥；这是 AML 调用本服务使用的 Memory System Key，不是 AML 发放的 Eval Key。
- `OPENAI_API_KEY` / `OPENAI_BASE_URL`：画像 GPT-4o-mini 使用的 CloseAI 中转站密钥和兼容地址。本次提交配置的地址为 `https://api.openai-proxy.org/v1`；密钥通过部署环境注入。
- `QIAN_WEN_API_KEY`：DashScope text-embedding-v4 密钥。
- `DEEPSEEK_API_KEY`：DS 官方 reranker 密钥。默认官方地址 `https://api.deepseek.com`。
- `MEMORY_PUBLIC_HOST`：正式部署时填写已解析到服务器的公网 DNS 域名。

不要把填好密钥的 `.env` 重新打进源代码包。部署包中的变量引用由 dotenv/Compose 解析，默认不会复制或打印密钥。

### 向 AML 主办方披露的模型调用渠道

本团队的画像组件通过 **CloseAI 第三方中转站** 调用声明为 `gpt-4o-mini` 的模型。下表是本次提交版本的提供方与请求配置，供主办方审核：

| 组件 | 调用渠道 | API Base URL | 请求模型标识 |
| --- | --- | --- | --- |
| 画像提取 | **CloseAI 中转站**，OpenAI-compatible API | `https://api.openai-proxy.org/v1` | `gpt-4o-mini` |
| Embedding | 阿里云 DashScope 官方 API | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `text-embedding-v4` |
| Reranker | DeepSeek 官方 API | `https://api.deepseek.com` | `deepseek-chat` |

画像请求发送至 `POST https://api.openai-proxy.org/v1/chat/completions`，请求中的 `model` 为 `gpt-4o-mini`，使用 CloseAI 提供的访问凭证。该渠道是第三方转发，不是直接连接 OpenAI 官方 API；实际转发及上游模型服务由 CloseAI 提供。此处披露的是参赛方使用的渠道和请求模型配置，不构成对中转站内部路由的独立验证，也不代表 AML 已批准该调用方式。模型提供方变更应同步更新提交说明，并由主办方审核。

CloseAI 仅用于本服务的 GPT-4o-mini 画像调用。DS reranker 仍直连 DeepSeek 官方服务；本包不包含 Answer/Judge 模型调用。密钥和鉴权头不随公开材料提交。

## 2. 本机启动

在解压目录运行，要求 Docker Desktop / Docker Engine 和 Compose v2：

```powershell
docker compose -f compose.yaml -f compose.local.yaml up --build -d
curl.exe http://127.0.0.1:8000/health
```

也可以使用 Python 3.12 和 Hatch：`hatch run runtime:python -m deployment.server`。
非 Docker 运行时请先设置 `SQLITE_PATH=state/memory.sqlite3`；Docker 默认存储在独立卷 `/data/memory.sqlite3`。
配置检查：`hatch run runtime:python -m deployment.server --check-config`，仅输出模型与功能开关，不输出密钥。

## 3. 公网 HTTPS 部署

在 Linux 服务器安装 Docker，解压并填写 `.env`，将 DNS 指向服务器，开放 TCP 80/443：

```sh
docker compose -f compose.yaml -f compose.https.yaml up --build -d
```

Caddy 自动申请 HTTPS 证书。该配置将后端保留在 Docker 内部网络，代理等待响应最多 31 分钟。不要同时叠加本机端口配置。

向 AML 填报：

| 项目 | 值 |
| --- | --- |
| Track / Division | Textual / Academic |
| Add URL | `https://你的域名/add` |
| Search URL | `https://你的域名/search` |
| Health URL | `https://你的域名/health`，GET 无需鉴权 |
| Authentication | `X-Api-Key: CHALLENGE_API_KEY的实际值`；也支持 Bearer、Token |
| Add completion | 同步 HTTP 200，持久化与向量写入完成后才返回 |
| Add / Search concurrency | 起步均填 1；仅单实例、单 Uvicorn worker |

公网 DNS/TLS、部署主机资源、外部模型配额和 Full 容量必须在目标服务器实测。本地容器检查不等于官方 smoke 或 Full 通过。不要把“起步并发 1”描述为已通过 Full 压力测试。

## 4. 模型与方法冻结

本包是 Academic 配置：画像 `gpt-4o-mini`，embedding `text-embedding-v4`，reranker 独立 `deepseek-chat`。启动入口会检查前两项，关闭未审计的可选 LLM 实验模块；DS 不参与画像生成。回答和评分由 AML 平台执行。

启用来源记录索引：长文本按来源边界组织检索片段，同时在 SQLite 保留完整原文和跨度映射；画像分批提取并校准来源索引。画像连续校验失败时，该批只保留原始内容，不生成无依据画像。外部服务失败仍返回非 2xx，不伪造写入成功。

现有边界切分器不会切开完整代码块/JSON 等不可分单元。单个不可分单元或 query 超过 24,000 UTF-8 字节时明确返回 422；不会静默截断或只写入前半段。普通长文可由多个合规来源片段组成。模型配额、限流、上游故障仍可能让请求失败。

详细方法、来源及本次部署变化见 `METHOD.md` / `NOTICE`。`MANIFEST.json` 的文件哈希定义本包快照；提交前填写 `SUBMISSION.yaml` 并发布对应固定代码版本。

## 5. 运维与数据保留

使用专属空卷启动正式评测；不要导入本地 benchmark 历史或答案。服务不记录请求正文，默认关闭访问日志。基础设施层也不要记录评测正文或鉴权头。数据库只用于评测、限制访问，并在评测结束后 30 天内删除相关数据及备份。

`docker compose down` 停止服务但保留卷；确认不再需要全部数据后，`docker compose down --volumes` 会删除本部署的记忆数据库及相关命名卷，不能用于清理仍在运行的评测。

SQLite 是单节点持久化方案，向量召回按用户扫描，Full 数据量下的性能需实测。不要共享同一个数据库文件给多台副本，不要在评测中途更换模型或方法。

## 官方资料（核对于 2026-10-26）

- [AML 文档：模型要求与自托管流程](https://agentmemories.ai/docs)
- [AML Add/Search 契约](https://agentmemories.ai/api-guide)
- [AML 参评说明](https://agentmemoryleaderboard.ai/competition/)

AML 要求参赛方提供可访问的已部署 API；提交 ZIP 或 Dockerfile 本身不会由 AML 自动部署。完成公开地址、署名材料与版本绑定后，使用 AML 发放的 Eval Key 运行官方 smoke，再申请 Full。本包不声称已经获得 AML 审核认证。
