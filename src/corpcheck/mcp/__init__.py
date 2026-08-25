"""MCP (Model Context Protocol) server exposing CorpCheck retrieval to any MCP client.

The server is a thin protocol adapter, not a second retrieval implementation. Every
tool routes through :func:`corpcheck.retrieval.pipeline.retrieve`, the same entry
point the HTTP API and the offline IR evaluation harness use, so what a Claude Code
session sees is exactly what ``evaluation/ir_eval.py`` measures. If those two ever
diverge, the evaluation numbers stop describing the product.

Run it with ``python -m corpcheck.mcp`` or the ``corpcheck-mcp`` console script.

中文：MCP 只适配协议和补充溯源信息；所有工具共用 HTTP 与评估使用的检索入口，避免形成第二套排序逻辑。
"""

from corpcheck.mcp.server import build_server, main

__all__ = ["build_server", "main"]
