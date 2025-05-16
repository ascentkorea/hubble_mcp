# builtin modules
import os
import json
from datetime import datetime, UTC
from typing import Union, Optional, Literal, List, Dict, Any
import asyncio
from functools import wraps

# installed modules
import httpx
from pydantic import BaseModel, Field, conlist
from mcp.server.fastmcp import FastMCP

HUBBLE_API_KEY = os.environ.get('HUBBLE_API_KEY')
assert HUBBLE_API_KEY, f"HUBBLE_API_KEY is not set"


class TooManyTriesException(Exception):
    pass


def async_retry(exceptions=(Exception), tries=3, delay=0.3, logger=None):

    def wrapper(func):

        @wraps(func)
        async def wrapped(*args, **kwargs):
            Tries = []
            for i in range(tries):
                try:
                    return await func(*args, **kwargs)
                except exceptions as ex:
                    ex_msg = f"Tries({ex.__class__.__name__}) Cnt: {i+1}, {ex}"
                    Tries.append(ex_msg)
                    if logger:
                        logger.warning(ex_msg)
                    else:
                        print(ex_msg)
                    if delay:
                        await asyncio.sleep(delay)

            raise TooManyTriesException(Tries)

        return wrapped

    return wrapper


class KeywordParameters(BaseModel):
    keywords: List = Field(
        min_items=1,
        max_items=1000,
        title="keyword(list)",
        description="요청 키워드",
    )
    '''
    gl: Geolocation of end user.
        https://www.link-assistant.com/news/googles-gl-parameter-explained.html
        https://developers.google.com/custom-search/v1/reference/rest/v1/cse/list#:~:text=Geolocation%20of%20end%20user.   
    '''
    gl: Literal['kr', 'jp'] = Field(
        default='kr',
        title="Geolocation",
        description="국가 코드",
    )
    _request_at: str
    _api_key: str

    def __init__(self, **data):
        super().__init__(**data)
        self._request_at = datetime.now(UTC).isoformat(timespec='milliseconds') + 'Z' # yapf:disable

class AdsMetrics(BaseModel):
    # https://developers.google.com/google-ads/api/reference/rpc/v17/KeywordPlanCompetitionLevelEnum.KeywordPlanCompetitionLevel
    competition: Optional[str] = Field(default=None,
                                       description="""
    competition_index 범위에 따른 구분값
    * LOW: 0~33
    * MEDIUM: 34~66
    * HIGH: 67~100""")
    competition_index: Optional[int] = Field(
        default=None, description="구글 검색광고 경쟁도 (구글 검색광고 슬롯 판매 비율)")
    cpc: Optional[float] = Field(default=None)
    low_bid_micros: Optional[int] = Field(default=None)
    high_bid_micros: Optional[int] = Field(default=None)
    volume_avg: Optional[int] = Field(description="최근 3개월의 월별 검색량의 평균값")
    volume_total: Optional[int] = Field(description="최근 12개월의 월별 검색량의 합산값")
    volume_trend: Optional[float] = Field(
        description="3개월 전 대비 최근 월별 검색량의 증감율")
    gg_volume_avg: Optional[int] = Field(description="(구글)최근 3개월의 월별 검색량의 평균값")
    gg_volume_total: Optional[int] = Field(
        description="(구글)최근 12개월의 월별 검색량의 합산값")
    gg_volume_trend: Optional[float] = Field(
        description="(구글)3개월 전 대비 최근 월별 검색량의 증감율")
    nv_volume_avg: Optional[int] = Field(
        description="(네이버)최근 3개월의 월별 검색량의 평균값")
    nv_volume_total: Optional[int] = Field(
        description="(네이버)최근 12개월의 월별 검색량의 합산값")
    nv_volume_trend: Optional[float] = Field(
        description="(네이버)3개월 전 대비 최근 월별 검색량의 증감율")

class KeywordInfo(BaseModel):
    keyword: str
    ads_metrics: Optional[AdsMetrics] = Field(default=None)
    features: Optional[object] = Field(default=None,
                                       description="""
해당 키워드의 SERP 에서
존재하는 타입별 갯수를 제공합니다.
> https://kr.listeningmind.com/hubble-guide/filter-serp-type/

존재하는 타입 종류는 아래와 같습니다.
* f_ad_bottom
* f_ad_top
* f_ads_bottom
* f_ads_top
* f_adult_certification
* f_app_results
* f_articles
* f_featured_snippet
* f_featured_snippet_ordered_list
* f_featured_snippet_table
* f_featured_snippet_unordered_list
* f_featured_snippet_video_exist
* f_google_play
* f_images
* f_job_search
* f_knowledge_panel
* f_local_results
* f_organic_results
* f_organic_results_rating
* f_organic_results_video
* f_others
* f_people_also_ask_for
* f_people_also_search_for
* f_product_comparison
* f_related_searches
* f_safe_search
* f_scholar
* f_sitelinks
* f_sns
* f_spell_check
* f_sticky_header_tabs
* f_twitter
* f_unit_converter
* f_unknown
* f_video_carousels
* f_video_results
    """)

    intents: Optional[object] = Field(default=None,
                                      description="""
검색 인텐트(https://www.ascentkorea.com/what-is-search-intent/)

* I: Informational
* N: Navigational
* C: Commercial
* T: Transactional
    """)
    demography: Optional[object] = Field(default=None,
                                         description="""
검색 유저 특성(https://kr.listeningmind.com/hubble-guide/filter-user/)

* m_gender: 남성 키워드 여부
* f_gender: 여성 키워드 여부
* a0: 12세 이하 키워드 여부
* a13: 13~19세 이하 키워드 여부
* a20: 20~24세 이하 키워드 여부
* a25: 25~29세 이하 키워드 여부
* a30: 30~39세 이하 키워드 여부
* a40: 40~49세 이하 키워드 여부
* a50: 50세 이상 키워드 여부
* m_gender_ratio: 남성 비율
* f_gender_ratio: 여성 비율
* a0_ratio: 12세 이하 비율
* a13_ratio: 13~19세 이하 비율
* a20_ratio: 20~24세 이하 비율
* a25_ratio: 25~29세 이하 비율
* a30_ratio: 30~39세 이하 비율
* a40_ratio: 40~49세 이하 비율
* a50_ratio: 50세 이상 비율
    """)
    monthly_volume: Optional[List[object]] = Field(default=None,
                                                   description="""
월별 검색량

* month: 해당 월
* gg: 구글 검색량
* nv: 네이버 검색량
* total: 구글 + 네이버 검색량
    """)

class KeywordData(BaseModel):
    infos: Optional[List[KeywordInfo]]

class BaseResponse(BaseModel):
    result: Literal["OK", "FAILED"] = Field(
        default="OK",
        title="성공 여부",
    )
    reason: str = Field(
        default="SUCCESS",
        title="실패시 추가 설명",
    )
    source: str
    version: str
    request_at: str = Field(
        default=None,
        title="요청시간",
        description="UTC 기준",
    )

class KeywordResponse(BaseResponse):
    request_detail: KeywordParameters = Field(description="요청 받았던 파라미터")
    cost: int = Field(default=0)
    remain_credits: int = Field(default=-1)
    data: Optional[KeywordData] = Field(default=None)

class ClusterParameters(BaseModel):
    keyword: str = Field(
        min_length=1,
        title="keyword(str)",
        description="요청 키워드",
    )
    gl: Literal['kr', 'jp'] = Field(
        default='kr',
        title="Geolocation",
        description="국가 코드",
    )
    limit: int = Field(default=1000, ge=1, le=10000, title='limit(int)', description='관계수 limit 욥션')
    hop: int = Field(
        default=2,
        ge=1,
        le=3,
        title="hop(int)",
        description="hop 수",
    )
    orientation: Literal['UNDIRECTED', 'NATURAL', 'REVERSE'] = Field(
        default='UNDIRECTED',
        title="direction",
        description="관계 방향",
    )
    _request_at: str
    _api_key: str

    def __init__(self, **data):
        super().__init__(**data)
        self._request_at = datetime.now(UTC).isoformat(timespec='milliseconds') + 'Z' # yapf:disable

class ClusterRels(BaseModel):
    closeness: int = Field(description="""
해당 type 에서 키워드가 출현한 위치
    """)
    distance: int = Field(description="""
전체 type 에서 키워드가 출현한 위치
    """)
    source: str
    target: str
    type: str = Field(description="""
관계 type

* PEOPLE_ALSO_SEARCH_FOR
* RELATED_SEARCHES
* REFINEMENTS
* PEOPLE_ALSO_ASK_FOR  
    """)

class ClusterData(BaseModel):
    nodes: List[str] = Field(description="키워드(노드) 리스트")
    nodes_count: int = Field(description="키워드(노드) 수")
    rels: List[ClusterRels]
    rels_count: int = Field(description="관계 수")

class ClusterResponse(BaseResponse):
    request_detail: ClusterParameters = Field(description="요청 받았던 파라미터")
    cost: int = Field(default=0)
    remain_credits: int = Field(default=-1)  # serviceAPI
    data: Optional[ClusterData] = Field(default=None)

class SerpParameters(BaseModel):
    """SERP 요청 형식
    """
    keyword: str = Field(min_length=1, title="요청 키워드")
    gl: Optional[Literal["kr", "jp", "us"]] = Field(
        default="kr",
        title="geo location",
    )
    num: Optional[Literal[10, 20]] = Field(
        default=20,
        title="result num",
        description="SERP 에 표출할 결과 개수",
    )
    device: Optional[Literal["mobile"]] = Field(
        default="mobile",
        title="device",
        description="(Deprecated) 요청 환경. mobile만 지원합니다.",
    )
    _request_at: str
    _api_key: str

    def __init__(self, **data):
        super().__init__(**data)
        self._request_at = datetime.now(UTC).isoformat(timespec='milliseconds') + 'Z' # yapf:disable

class SerpResponse(BaseResponse):
    """SERP 응답 형식
    """
    request_detail: SerpParameters = Field(description="요청 받았던 파라미터")
    cost: int = Field(default=0, title="", description="")
    remain_credits: int = Field(default=-1, title="", description="")
    data: Optional[List[dict]] = Field(default=None, title="", description="")



# Initialize FastMCP server
mcp = FastMCP("hubble")

@mcp.tool()
@async_retry(exceptions=(Exception), tries=2, delay=0.3)
async def get_search_path(keyword: str,
                          gl: Literal['kr', 'jp'] = "kr",
                          limit=200) -> dict[str, Any] | None:
    """Make a request to the Path Finder API of Hubble with proper error handling.
        Args:
            keyword: str, 검색 키워드
            gl: str, 지역 코드 한국 일본(kr, jp)
            limit: int, 검색 경로 분석 결과 최대 개수(기본값 200)
        Returns:
            dict[str, Any] | None: 검색 경로 분석 결과
    """

    payload = {"keyword": keyword, "gl": gl, "limit": limit}
    headers = {"X-API-Key": HUBBLE_API_KEY}
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://hubble-data-api.ascentlab.io/path_finder",
            headers=headers,
            json=payload,
            timeout=30.0)
        response.raise_for_status()
        return response.text


@mcp.tool()
@async_retry(exceptions=(Exception), tries=2, delay=0.3)
async def get_keyword_info(
        req_param: KeywordParameters) -> dict[KeywordResponse, Any] | None:
    '''
        키워드 정보 조회 (최대 1000개 키워드 조회 가능)
        키워드 정보 조회 결과는 키워드 정보 조회 결과 형식에 따라 반환됩니다.
        args:
            req_param: KeywordParameters, 키워드 정보 조회 요청 파라미터
        returns:
            dict[KeywordResponse, Any] | None: 키워드 정보 조회 결과
    KeywordResponse 는 아래와 같은 정보를 포함합니다:

    ads_metrics:
        - competition: 경쟁도 수준 (LOW: 0-33, MEDIUM: 34-66, HIGH: 67-100)
        - competition_index: 구글 검색광고 경쟁도 (0-100)
        - cpc: 클릭당 비용
        - volume_avg: 최근 3개월 월평균 검색량
        - volume_total: 최근 12개월 총 검색량
        - volume_trend: 3개월 전 대비 검색량 증감율
        - gg_volume_avg: 구글 최근 3개월 월평균 검색량
        - gg_volume_total: 구글 최근 12개월 총 검색량 
        - gg_volume_trend: 구글 검색량 증감율
        - nv_volume_avg: 네이버 최근 3개월 월평균 검색량
        - nv_volume_total: 네이버 최근 12개월 총 검색량
        - nv_volume_trend: 네이버 검색량 증감율


    intents:
        - I: Informational
        - N: Navigational
        - C: Commercial
        - T: Transactional


    monthly_volume:
        - month: 해당 월
        - gg: 구글 검색량
        - nv: 네이버 검색량
        - total: 구글 + 네이버 검색량
    
    '''

    async with httpx.AsyncClient() as client:
        req_param_json = req_param.model_dump_json()
        headers = {"X-API-Key": HUBBLE_API_KEY}
        response = await client.post(
            "https://hubble-data-api.ascentlab.io/keyword",
            headers=headers,
            json=json.loads(req_param_json),
            timeout=30.0)
        response.raise_for_status()
        return response.text

@mcp.tool()
@async_retry(exceptions=(Exception), tries=2, delay=0.3)
async def get_graph_info(
        req_param: ClusterParameters) -> dict[ClusterResponse, Any] | None:
    '''
    키워드 관계 정보(허블의 클러스터 파인더 결과 조회, 키워드 관계 정보 조회)
    args:
        req_param: ClusterParameters, 키워드 관계 정보 조회 요청 파라미터
    returns:
        dict[ClusterResponse, Any] | None: 키워드 관계 정보 조회 결과
    ClusterResponse 는 아래와 같은 정보를 포함합니다:  
    
    nodes: 키워드(노드) 리스트
    nodes_count: 키워드(노드) 수
    rels: 관계 리스트
    rels_count: 관계 수

    closeness: 관계에서 키워드가 출현한 위치.
    distance: 모든 관계에서 키워드가 출현한 위치.
    type: PEOPLE_ALSO_SEARCH_FOR | RELATED_SEARCHES | REFINEMENTS | PEOPLE_ALSO_ASK_FOR  

    '''

    async with httpx.AsyncClient() as client:
        req_param_json = req_param.model_dump_json()
        headers = {"X-API-Key": HUBBLE_API_KEY}
        response = await client.post(
            "https://hubble-data-api.ascentlab.io/cluster",
            headers=headers,
            json=json.loads(req_param_json),
            timeout=30.0)
        response.raise_for_status()
        return response.text

@mcp.tool()
@async_retry(exceptions=(Exception), tries=2, delay=0.3)
async def crawl_google_serp(
        keyword: str,
        gl: Literal['kr', 'us', 'jp'] = "kr") -> dict[SerpResponse, Any] | None:
    '''
    구글 SERP API 요청
    '''

    async with httpx.AsyncClient() as client:
        payload = {
            "keyword": keyword,
            "gl": gl
        }
        headers = {"X-API-Key": HUBBLE_API_KEY}
        response = await client.post(
            "https://hubble-data-api.ascentlab.io/serp",
            headers=headers,
            json=payload,
            timeout=30.0)
        response.raise_for_status()
        return response.text

@mcp.tool()
@async_retry(exceptions=(Exception), tries=2, delay=0.3)
async def crawl_web_page(
        url_list: List[str]) -> dict[Any] | None:
    '''
    웹 페이지 크롤링
    args:
        url_list: List[str], 크롤링할 웹 페이지 리스트
    returns:
        dict[Any] | None: 크롤링 결과
    '''

    async with httpx.AsyncClient() as client:
        headers = {"X-API-Key": HUBBLE_API_KEY}
        response = await client.post(
            "https://hubble-data-api.ascentlab.io/web_crawl",
            headers=headers,
            json=url_list,
            timeout=30.0)
        response.raise_for_status()
        return response.text
    

@mcp.tool()
@async_retry(exceptions=(Exception), tries=2, delay=0.3)
async def crawl_google_suggest_extensions(
        keyword: str,
        gl: Literal['kr', 'us', 'jp'] = "kr") -> dict[Any] | None:
    '''
    키워드에 대한 구글 서제스트 확장 키워드 수집 요청
    args:
        keyword: str, 키워드
        gl: Literal['kr', 'us', 'jp'] = "kr", 국가 코드
    returns:
        dict[Any] | None: 구글 서제스트 확장 키워드 수집 결과

    키워드 확장
    suggestions(common)
    키워드의 suggestions(1 depth) + 1 depth 결과들의 suggestions(2 depth)

    suffix(나라별)

    한국 (gl : kr) : 2405개
    keyword + space (1 개)
    keyword + 한글 자음(ㄱ~ㅎ: 19 개)
    keyword + 한글 음절 (2350 개)
    keyword + 알파벳(a~z: 26 게)
    keyword + 숫자(0~9: 10 개)

    미국 (gl : us) : 37개
    keyword + space (1 개)
    keyword + 알파벳(a~z: 26 개)
    keyword + 숫자(0~9: 10 개)

    일본 (gl : jp) : 2318개
    keyword + space (1 개)
    keyword + 히라가나 (46 개)
    keyword + 히라가나 * 히라가나 (2116 개)
    keyword + 가타카나 (46 개)
    keyword + 일어 음절 (73 개)
    keyword + 알파벳(a~z: 26 개)
    keyword + 숫자(0~9: 10 개)
    '''

    async with httpx.AsyncClient() as client:
        payload = {
            "keyword": keyword,
            "gl": gl
        }
        headers = {"X-API-Key": HUBBLE_API_KEY}
        response = await client.post(
            "https://hubble-data-api.ascentlab.io/google_suggest_ex",
            headers=headers,
            json=payload,
            timeout=30.0)
        response.raise_for_status()
        return response.text
    

@mcp.tool()
@async_retry(exceptions=(Exception), tries=2, delay=0.3)
async def crawl_google_trends(
        keywords: List[str],
        location: Literal['South Korea', 'Japan'],
        timeframe: Literal['now 1-H', 'now 7-d', 'today 1-m', ],
        gl: Literal['kr', 'jp']) -> dict[Any] | None:
    '''
    구글 트렌드 수집 요청
    최근 며칠 이내의 키워드 트렌드 추이를 0~100 사이의 값으로 표현 됩니다.
    (검색량은 아니고, 검색 관심도를 나타냅니다. 해당 수치는 0~100 사이의 값으로 표현 됩니다.)
    trends: 기간을 기준으로 차트에서 가장 높은 지점 대비 검색 관심도를 나타냅니다. 
    값은 검색 빈도가 가장 높은 검색어의 경우 100, 검색 빈도가 그 절반 정도인 검색어의 경우 50, 
    해당 검색어에 대한 데이터가 충분하지 않은 경우 0으로 나타납니다.

    args:
        keywords: List[str], 키워드 리스트
        location: Literal['South Korea', 'Japan'],
        timeframe: Literal['now 1-H', 'now 7-d', 'today 1-m'],
        gl: Literal['kr', 'jp']
    returns:
        dict[Any] | None: 구글 트렌드 수집 결과
    '''

    async with httpx.AsyncClient() as client:
        payload = {
            "keywords": keywords,
            "gl": gl,
            "location": location,
            "timeframe": timeframe
        }
        headers = {"X-API-Key": HUBBLE_API_KEY}
        response = await client.post(
            "https://hubble-data-api.ascentlab.io/google_trend",
            headers=headers,
            json=payload,
            timeout=30.0)
        response.raise_for_status()
        return response.text




if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')


    # print("...")
    # resp = asyncio.run(get_alerts("CA"))
    # print(resp)
    # resp = asyncio.run(get_search_path("냉장고"))
    # print(resp)
    # req_param = KeywordParameters(keywords=["냉장고"])
    # resp = asyncio.run(get_keyword_info(req_param))
    # print(resp)
    # req_param = ClusterParameters(keyword="냉장고", limit=20, hop=2, orientation="UNDIRECTED")
    # resp = asyncio.run(get_graph_info(req_param))
    # print(resp)
    # req_param = SerpParameters(keyword="냉장고")
    # resp = asyncio.run(get_serp_info(req_param))
    # print(resp)
    # url_list = ["https://www.ascentkorea.com/seo_six_essential_elements/","https://www.ascentkorea.com/about/"]
    # resp = asyncio.run(crawl_web_page(url_list))
    # print(resp)
    # resp = asyncio.run(crawl_google_suggest_extensions("냉장고", "kr"))
    # print(resp)
    # resp = asyncio.run(crawl_google_trends(["냉장고"], "South Korea", "now 7-d", "kr"))
    # print(resp)
