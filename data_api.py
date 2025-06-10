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

HUBBLE_API_URL = "https://listeningmind-mcp-api.ascentlab.io"

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
        default=None, description="구글 검색광고 경쟁도 (검색광고 슬롯 판매 비율)")
    cpc: Optional[float] = Field(default=None)
    low_bid_micros: Optional[int] = Field(default=None)
    high_bid_micros: Optional[int] = Field(default=None)
    volume_avg: Optional[int] = Field(description="최근 3개월의 월별 검색량 평균값")
    volume_total: Optional[int] = Field(description="최근 12개월의 월별 검색량 합산값")
    volume_trend: Optional[float] = Field(
        description="3개월 전 대비 최근 월 검색량 증감율")
    gg_volume_avg: Optional[int] = Field(description="(구글)최근 3개월의 월 검색량 평균값")
    gg_volume_total: Optional[int] = Field(
        description="(구글)최근 12개월의 월 검색량 합산값")
    gg_volume_trend: Optional[float] = Field(
        description="(구글)3개월 전 대비 최근 월 검색량 증감율")
    nv_volume_avg: Optional[int] = Field(
        description="(네이버)최근 3개월의 월 검색량 평균값")
    nv_volume_total: Optional[int] = Field(
        description="(네이버)최근 12개월의 월 검색량 합산값")
    nv_volume_trend: Optional[float] = Field(
        description="(네이버)3개월 전 대비 최근 월 검색량 증감율")
class KeywordInfo(BaseModel):
    keyword: str
    ads_metrics: Optional[AdsMetrics] = Field(default=None)
    features: Optional[object] = Field(default=None,
                                       description="""
해당 키워드의 검색 결과 페이지(SERP)에 어떤 검색 스니펫(아래 리스트 참조)이 노출 되고 있는 지를 제공합니다.
> https://kr.listeningmind.com/hubble-guide/filter-serp-type/
존재하는 검색 스니펫 타입 종류는 아래와 같습니다.
* f_ad_bottom: 검색 결과 하단에 존재하는 광고 (해당 키워드가 상업형 혹은 거래형 키워드일 가능성을 보여줌)
* f_ad_top: 검색 결과 상단에 존재하는 광고 (해당 키워드가 상업형 혹은 거래형 키워드일 가능성을 보여줌)
* f_ads_bottom: 검색 결과 하단에 존재하는 광고 (해당 키워드가 상업형 혹은 거래형 키워드일 가능성을 보여줌)
* f_ads_top: 검색 결과 상단에 존재하는 광고 (해당 키워드가 상업형 혹은 거래형 키워드일 가능성을 보여줌)
* f_adult_certification: 성인 인증이 필요한 검색 결과
* f_app_results: 앱 검색 결과
* f_articles: 뉴스 기사 검색 결과 (해당 키워드가 뉴스성이 있는 키워드임을 보여줌)
* f_featured_snippet: 구글의 피쳐드 스니펫
* f_featured_snippet_ordered_list: 구글의 피쳐드 스니펫 중 순서가 있는 리스트
* f_featured_snippet_table: 구글의 피쳐드 스니펫 중 표 형태로 제공되는 리스트
* f_featured_snippet_unordered_list: 구글의 피쳐드 스니펫 중 순서가 없는 리스트
* f_featured_snippet_video_exist: 구글의 피쳐드 스니펫 중 비디오가 포함된 리스트
* f_google_play: 구글 플레이 검색 결과 (해당 키워드가 앱과 관련된 키워드임을 보여줌)
* f_images: 이미지 검색 결과 (해당 키워드가 이미지 검색 결과가 있는 키워드임을 보여줌)
* f_job_search: 구글의 구인 검색 결과 (해당 키워드가 구인과 관련된 키워드임을 보여줌)
* f_knowledge_panel: 구글의 지식 패널 검색 결과 (해당 키워드가 브랜드나 인물 등과 관련된 키워드임을 보여줌)
* f_local_results: 구글의 로컬 검색 결과 (해당 키워드가 지역과 관련된 키워드임을 보여줌)
* f_organic_results: 구글의 유기적 검색 결과 
* f_organic_results_rating: 구글의 유기적 검색 결과 중 평점이 있는 검색 결과 (해당 키워드의 제품이나 브랜드가 리뷰나 평점을 받고 있음을 보여줌)
* f_organic_results_video: 구글의 유기적 검색 결과 중 비디오가 포함된 검색 결과 (해당 키워드가 비디오 포맷의 콘텐츠에 대한 니즈를 가질 가능성 혹은 유튜브에서도 검색되는 키워드임을 보여줌)
* f_others: 기타 검색 결과
* f_people_also_ask_for: 구글의 People Also Ask 검색 결과
* f_people_also_search_for: 구글의 People Also Search For 검색 결과
* f_product_comparison: 제품 비교 검색 결과
* f_related_searches: 구글의 Related Searches 검색 결과
* f_safe_search: 안전 검색이 적용된 검색 결과 (해당 키워드가 안전 검색이 적용된 키워드임을 보여줌)
* f_scholar: 구글 스칼라 검색 결과 (해당 키워드가 학술적이거나 연구 관련된 키워드임을 보여줌)
* f_sitelinks: 구글의 사이트 링크 검색 결과 (해당 키워드가 특정 웹사이트나 브랜드와 관련된 키워드임을 보여줌)
* f_sns: 소셜 미디어 검색 결과 (해당 키워드가 소셜 미디어에서 급상승 혹은 관련 맨션이 많은 키워드임을 보여줌)
* f_spell_check: 구글의 스펠링 검사 검색 결과 (해당 키워드가 오타 혹은 일반적으로 상요되는 다른 표현이 있는 키워드임을 보여줌)
* f_sticky_header_tabs: 구글의 스티키 헤더 탭 검색 결과 (해당 키워드가 검색 결과 상단에 탭 형태로 제공되는 키워드임을 보여줌)
* f_unit_converter: 단위 변환 검색 결과 
* f_unknown: 구글의 알수 없는 검색 결과
* f_video_carousels: 구글의 비디오 캐러셀 검색 결과 (해당 키워드가 비디오 포맷의 콘텐츠에 대한 니즈를 가질 가능성 혹은 유튜브에서도 검색되는 키워드임을 보여줌)
* f_video_results: 구글의 비디오 검색 결과 (해당 키워드가 비디오 포맷의 콘텐츠에 대한 니즈를 가질 가능성 혹은 유튜브에서도 검색되는 키워드임을 보여줌)
    """)
    intents: Optional[object] = Field(default=None,
                                      description="""
검색 인텐트(https://www.ascentkorea.com/what-is-search-intent/)
* I: 정보형(Informational)잠재 소비자가 제품 자체 정보나 제품이 포함된 카테고리 등에 대한 정보를 얻기 위한 검색
* N: 네비게이셔널형(Navigational)잠재 소비자가 특정 웹사이트나 브랜드 혹은 매장 위치를 찾기 위한 검색
* C: 상업형(Commercial) 구매 전 단계에서 비교, 리뷰, 추천 등의 구매 의사 결정에 도움이 되는 정보를 구하려는 목적의 검색
* T: 거래형(Transactional)구매를 목적으로 하는 검색
    """)
    demography: Optional[object] = Field(default=None,
                                         description="""
검색 유저 특성(https://kr.listeningmind.com/hubble-guide/filter-user/)
* m_gender: 표준편차 (시그마 1) 이상으로 남성이 더 많이 검색했을 가능성이 높은 키워드
* f_gender: 표준편차 (시그마 1) 이상으로 여성이 더 많이 검색했을 가능성이 높은 키워드
* a0: 표준편차 (시그마 1) 이상으로 12세 이하가 더 많이 검색했을 가능성이 높은 키워드
* a13:  표준편차 (시그마 1) 이상으로 13~19세가 더 많이 검색했을 가능성이 높은 키워드
* a20: 표준편차 (시그마 1) 이상으로 20~24세가 더 많이 검색했을 가능성이 높은 키워드
* a25: 표준편차 (시그마 1) 이상으로 25~29세가 더 많이 검색했을 가능성이 높은 키워드
* a30: 표준편차 (시그마 1) 이상으로 30~39세가 더 많이 검색했을 가능성이 높은 키워드
* a40: 표준편차 (시그마 1) 이상으로 40~49세가 더 많이 검색했을 가능성이 높은 키워드
* a50: 50세 이상이 더 많이 검색했을 가능성이 높은 키워드
* m_gender_ratio: 남성 검색자의 비율
* f_gender_ratio: 여성 검색자의 비율
* a0_ratio: 12세 이하 검색자의 비율
* a13_ratio: 13~19세 검색자의 비율  
* a20_ratio: 20~24세 검색자의 비율
* a25_ratio: 25~29세 검색자의 비율
* a30_ratio: 30~39세 검색자의 비율
* a40_ratio: 40~49세 검색자의 비율
* a50_ratio: 50세 이상 검색자의 비율    
    """)
    monthly_volume: Optional[List[object]] = Field(default=None,
                                                   description="""
월별 검색량
* month: 해당 월
* gg: 구글 검색량
* nv: 네이버 검색량
* total: 구글 검색량 + 네이버 검색량
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
                          limit=300) -> dict[str, Any] | None:
    """Make a request to the Path Finder API of Hubble with proper error handling.
        Args:
            keyword: str, 검색 키워드(모든 키워드는 소문자로 변환하여 요청)
            gl: str, 지역 코드 한국 일본(kr, jp)
            limit: int, 검색 경로 분석 결과 최대 개수(기본값 300)
        Returns:
            dict[str, Any] | None: 검색 키워드의 검색 경로 분석 결과
    """
    payload = {"keyword": keyword, "gl": gl, "limit": limit}
    headers = {"X-API-Key": HUBBLE_API_KEY}
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{HUBBLE_API_URL}/path_finder",
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
        모든 키워드는 소문자로 변환하여 요청
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
        - I: Informational - 잠재 소비자가 제품 자체 정보나 제품이 포함된 카테고리 등에 대한 정보를 얻기 위한 검색
        - N: Navigational - 잠재 소비자가 특정 웹사이트나 브랜드 혹은 매장 위치를 찾기 위한 검색
        - C: Commercial - 구매 전 단계에서 비교, 리뷰, 추천 등의 구매 의사 결정에 도움이 되는 정보를 구하려는 목적의 검색
        - T: Transactional - 구매를 목적으로 하는 검색
    monthly_volume:
        - month: 해당 월
        - gg: 구글 검색량
        - nv: 네이버 검색량
        - total: 구글 + 네이버 검색량
    
    '''
    async with httpx.AsyncClient() as client:
        headers = {"X-API-Key": HUBBLE_API_KEY}
        response = await client.post(
            f"{HUBBLE_API_URL}/keyword",
            headers=headers,
            json=req_param.model_dump(),
            timeout=30.0)
        response.raise_for_status()
        return response.text
@mcp.tool()
@async_retry(exceptions=(Exception), tries=2, delay=0.3)
async def get_graph_info(
        req_param: ClusterParameters) -> dict[ClusterResponse, Any] | None:
    '''
    키워드 관계 정보(리스닝마인드의 클러스터 파인더의 결과 조회, 키워드 관계 정보 조회)
    모든 키워드는 소문자로 변환하여 요청
    args:
        req_param: ClusterParameters, 키워드 관계 정보 조회 요청 파라미터
    returns:
        dict[ClusterResponse, Any] | None: 키워드 관계 정보 조회 결과
    ClusterResponse 는 아래와 같은 정보를 포함합니다:  
    
    nodes: 조회한 키워드의 앞과 뒤로 2혹은 2hop 거리 안에서 검색된 모든 키워드(노드) 리스트
    nodes_count: 키워드(노드) 수
    rels: 관계 리스트
    rels_count: 관계 수
    closeness: 관계에서 키워드가 출현한 위치.
    distance: 모든 관계에서 키워드가 출현한 위치.
    type: PEOPLE_ALSO_SEARCH_FOR | RELATED_SEARCHES | REFINEMENTS | PEOPLE_ALSO_ASK_FOR  
    '''
    async with httpx.AsyncClient() as client:
        headers = {"X-API-Key": HUBBLE_API_KEY}
        response = await client.post(
            f"{HUBBLE_API_URL}/cluster",
            headers=headers,
            json=req_param.model_dump(),
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
            f"{HUBBLE_API_URL}/google_serp",
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
            f"{HUBBLE_API_URL}/web_crawl",
            headers=headers,
            json=url_list,
            timeout=30.0)
        response.raise_for_status()
        return response.text
    
@mcp.tool()
@async_retry(exceptions=(Exception), tries=2, delay=0.3)
async def crawl_google_suggest(
        q: str,
        gl: Literal['kr', 'us', 'jp'] = "kr") -> dict[Any] | None:
    '''
    입력된 키워드에 대해 구글 서제스트에 나타난 키워드 수집 요청
    args:
        keyword: str, 키워드
        gl: Literal['kr', 'us', 'jp'] = "kr", 국가 코드
    returns:
        dict[Any] | None: 구글 서제스트 수집 결과
    키워드 suggestions
    한국 (gl: kr)
    미국 (gl: us)
    일본 (gl: jp)
    '''

    async with httpx.AsyncClient() as client:
        payload = {
            "q": q,
            "gl": gl,
        }
        headers = {"X-API-Key": HUBBLE_API_KEY}
        response = await client.post(
            f"{HUBBLE_API_URL}/google_suggest",
            headers=headers,
            json=payload,
            timeout=30.0)
        response.raise_for_status()
        return response.text
    
class GoogleTrendsParameters(BaseModel):
    keywords: List[str] = Field(
        min_items=1,
        max_items=3,
    )
    location: Literal['South Korea', 'Japan']
    timeframe: Literal['now 1-H', 'now 7-d', 'today 1-m']
    gl: Literal['kr', 'jp']
@mcp.tool()
@async_retry(exceptions=(Exception), tries=2, delay=0.3)
async def crawl_google_trends(
        req_param: GoogleTrendsParameters) -> dict[Any] | None:
    '''
    구글 트렌드 수집 요청
    최근 며칠 이내의 키워드 트렌드 추이를 0~100 사이의 값으로 표현 됩니다.
    (검색량은 아니고, 검색 관심도를 나타냅니다. 해당 수치는 0~100 사이의 값으로 표현 됩니다.)
    trends: 기간을 기준으로 차트에서 가장 높은 지점 대비 검색 관심도를 나타냅니다. 
    값은 검색 빈도가 가장 높은 검색어의 경우 100, 검색 빈도가 그 절반 정도인 검색어의 경우 50, 
    해당 검색어에 대한 데이터가 충분하지 않은 경우 0으로 나타납니다.
    키워드 하나에 대한 검색 관심도 추이를 알수 있으며, 최대 3개 키워드를 비교 할수 있습니다.
    특정 키워드하나를 입력했을때, 특정 기간의 최대값이 100 이라고 했을때,
    키워드 여러개 입력시에는 검색관심도가 가장 큰 키워드는 0~100 사이값으로 표현되고, 나머지는 적절히 스케일링 되므로 
    비교시에 특정 키워드의 최대값은 100이 아닐수 있습니다.
    따라서, 키워드간 관심도 비교시에 4개 이상의 키워드를 비교 하기 위해서는 
    우선 3개를 비교 하고, 이후 가장 높은 관심도가 있는 키워드를 계속 같이 추가해야 각 수치간에 비교가 가능해집니다.
    args:
        keywords: List[str], 키워드 리스트
        location: Literal['South Korea', 'Japan'],
        timeframe: Literal['now 1-H', 'now 7-d', 'today 1-m'],
        gl: Literal['kr', 'jp']
    returns:
        dict[Any] | None: 구글 트렌드 수집 결과
    '''
    async with httpx.AsyncClient() as client:
        headers = {"X-API-Key": HUBBLE_API_KEY}
        response = await client.post(
            f"{HUBBLE_API_URL}/google_trend",
            headers=headers,
            json=req_param.model_dump(),
            timeout=30.0)
        response.raise_for_status()
        return response.text


def main():
    # Initialize and run the server
    mcp.run(transport='stdio')


if __name__ == "__main__":
    main()


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
    # resp = asyncio.run(crawl_google_serp(keyword="섬유탈취제 언제 사용", gl="kr"))
    # print(resp)
    # url_list = ["https://www.ascentkorea.com/seo_six_essential_elements/","https://www.ascentkorea.com/about/"]
    # resp = asyncio.run(crawl_web_page(url_list))
    # print(resp)
    # resp = asyncio.run(crawl_google_suggest("냉장고", "kr"))
    # print(resp)
    # req_param = GoogleTrendsParameters(keywords=["냉장고"], location="South Korea", timeframe="now 7-d", gl="kr")
    # resp = asyncio.run(crawl_google_trends(req_param))
    # print(resp)