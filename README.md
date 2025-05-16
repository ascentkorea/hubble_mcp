# hubble_mcp

> ### ⚠️ **주의**
> 현재는 사내 VPN 환경에서만 사용 가능합니다.
> > 간혹 API 가 실패 할수 있으니, 데이터 응답을 정상적으로 수신 했는지 확인이 필요 합니다.  
> > (클로드가 툴 사용시 나온 메시지를 클릭해서 API 응답 메시지 확인이 가능합니다. )

## set-up-your-environment

* Claude Desktop 설치
  * https://claude.ai/download
* uv 설치 (hubble MCP Server 는 파이썬으로 작성되었습니다.) - https://modelcontextprotocol.io/quickstart/server#set-up-your-environment
  * Windows
    ```
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```
    일부 보안 프로그램의 실시간 감시에서 `uv` 명령어가 차단될 경우, 실시간 감시를 중단 또는 해당 파일을 제외 시켜 주세요~
  * MacOS/Linux
    ```
    curl -LsSf https://astral.sh/uv/install.sh | sh
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

Windows 설정 파일 샘플
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

MacOS/Linux 설정 파일 샘플
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

## 제공 되는 tool(기능) 설명

* get_search_path: 검색 경로 API
* get_keyword_info: 키워드에 대한 검색량및 다양한 정보
* get_graph_info: 키워드간의 관계 정보
* crawl_google_serp: 구글 SERP API
* crawl_google_suggest_extension: 구글 서제스트 API
  > ⚠️ 현재 응답데이터가 너무 많아서, 조절 하려고 하고 있습니다. 당분간 disable 추천 드립니다.
* crawl_google_trends: 구글 트렌드 API
