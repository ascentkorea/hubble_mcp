# hubble_mcp

## set-up-your-environment

* Claude Desktop 설치
  * https://claude.ai/download
* uv 설치 (hubble MCP Server 는 파이썬으로 작성되었습니다.)
  > 참조: https://modelcontextprotocol.io/quickstart/server#set-up-your-environment
  * MacOS/Linux
    ```
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
  * Windows
    ```
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```
  
* Hubble MCP Server 다운로드
  * https://github.com/ascentkorea/hubble_mcp/releases
    * zip 파일 다운로드후, 압축해제 (아래 설정파일 에서 해당 경로 필요)
    * 가장 최신 버전 사용
* HUBBLE_API_KEY 발급

## claude_desktop_config.json  파일 설정

* 클로드 데스크탑 > 파일 > 설정(Ctrl + ,) > 개발자 > 설정 편집
  * json 파일 수정 이후, 클로드 데스크탑 > 파일 > 종료 를 통해서 다시 시작 필요!
* 맥일경우 uv 실행 파일의 절대 경로 필요, 아래 맥 샘플 참조


```json
{
  "mcpServers": {
    "hubble": {
      "command": "uv",
      "args": [
          "--directory",
          "<Hubble MCP Server 다운로드후 zip 파일 압축 해제후 data_api.py 파일 있는 경로>",
          "run",
          "data_api.py"
      ],
      "env": {
        "HUBBLE_API_KEY": "<HUBBLE_API_KEY>"
      }
    }
  }
}
```

윈도우 샘플
```json
{
  "mcpServers": {
    "hubble": {
      "command": "uv",
      "args": [
          "--directory",
          "C:\\Users\\XXXX\\Documents\\hubble\\hubble_mcp",
          "run",
          "data_api.py"
      ],
      "env": {
        "HUBBLE_API_KEY": "xxx-xxxx-xxxx-xxx"
      }
    }
  }
}
```

맥 샘플
```json
{
  "mcpServers": {
    "hubble": {
      "command": "/Users/XXX/.local/bin/uv",
      "args": [
          "--directory",
          "/Users/XXX/workspace/hubble/hubble_mcp",
          "run",
          "data_api.py"
      ],
      "env": {
        "HUBBLE_API_KEY": "xxx-xxxx-xxxx-xxx"
      }
    }
  }
}
```

