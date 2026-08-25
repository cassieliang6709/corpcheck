"""``python -m corpcheck.mcp`` — start the stdio MCP server.

中文：命令行模块只委托给服务器入口；stdio 是 JSON-RPC 通道，业务日志由服务器写入 stderr。
"""

from corpcheck.mcp.server import main

if __name__ == "__main__":
    main()
