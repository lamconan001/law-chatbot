"""
Module preprocess văn bản luật: Convert, Clean, Chunking
"""
import re
from typing import Dict, List
from dataclasses import dataclass
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.openai import OpenAI
from prompts import VANBAN_SUMMARIZATION_PROMPT, CHUONG_SUMMARIZATION_PROMPT
from underthesea import word_tokenize
from storage_system import LegalMongoRepository, LegalQdrantRepository
from embedding import BGEEmbedding
import uuid

@dataclass
class VanBan:
    id: str #id là số hiện văn bản
    ten: str
    noi_dung: str
    tom_luoc: str = ""
    key_words: List[str] = None
    chuongs: List['Chuong'] = None
@dataclass
class Chuong:
    id: str
    van_ban_id: str
    so_chuong: str
    ten_chuong: str
    noi_dung: str
    tom_luoc: str =""
    key_words: List[str] = None
    dieus: List['Dieu'] = None
@dataclass
class Dieu:
    id: str
    chuong_id: str
    van_ban_id: str
    so_dieu: str
    ten_dieu: str
    noi_dung: str
    key_words: List[str] = None

# Convert và tiền xử lý văn bản luật
class LegalDocumentPreprocessor:
    """Xử lý tiền xử lý văn bản luật"""
    
    def __init__(self):
        pass

    def convert_document(self, file_path: str, format: str = 'pdf') -> str:
        """
        Convert văn bản từ nhiều định dạng về text
        Hỗ trợ: PDF, DOCX, DOC, HTML
        """
        if format == 'pdf':
            return self._convert_pdf(file_path)
        else:
            print("Chức năng convert chỉ hỗ trợ định dạng PDF hiện tại.")
            return ""
    def _convert_pdf(self, file_path: str) -> str:
        """
        Convert PDF về text
        """
        import pdfplumber
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text
    
    def preprocess_full_text(self, file_path:str, format: str = 'pdf') -> str:
        """
        Tiền xử lý văn bản luật: loại bỏ ký tự đặc biệt, chuẩn hóa khoảng trắng, chuẩn hóa Unicode
        """
        text = self.convert_document(file_path=file_path, format='pdf')
        return text.strip()

# Chunking văn bản luật thành chương, điều 
class LegalDocumentSplitter:
    """Chia nhỏ văn bản luật thành các đoạn nhỏ hơn dựa trên chương, điều, khoản."""
    
    def __init__(self):
        self.chuong_pattern = re.compile(r'Chương\s+([IVXLCM]+|[0-9]+)\.?\s*(.*)', re.IGNORECASE)
        self.dieu_pattern = re.compile(r'Điều\s+(\d+[A-Z]?)\.\s*(.+)', re.IGNORECASE)

    def split_into_chuongs(self, text: str, van_ban_id: str) -> List[Dict]:
        """
        Chia văn bản thành các chương.
        """
        chuongs = []
        # Tìm tất cả các chương
        chuong_matches = list(self.chuong_pattern.finditer(text))
        
        for i, match in enumerate(chuong_matches):
            so_chuong = match.group(1)
            ten_chuong = match.group(2).strip()
            start_pos = match.end()
            
            # Xác định vị trí kết thúc chương
            end_pos = chuong_matches[i + 1].start() if i + 1 < len(chuong_matches) else len(text)
            noi_dung = text[start_pos:end_pos].strip()
            
            chuong = Chuong(
                id=f"{van_ban_id}_C{so_chuong}",
                van_ban_id=van_ban_id,
                so_chuong=so_chuong,
                ten_chuong=ten_chuong,
                noi_dung=noi_dung
            )
            chuongs.append(chuong)

        return chuongs

    def split_into_dieus(self, chuong: Chuong) -> List[Dieu]:
        """Tách chương thành các điều"""
        dieus = []
        
        # Tìm tất cả các điều trong chương
        dieu_matches = list(self.dieu_pattern.finditer(chuong.noi_dung))
        
        for i, match in enumerate(dieu_matches):
            so_dieu = match.group(1)
            ten_dieu = match.group(2).strip()
            start_pos = match.end()
            
            # Xác định vị trí kết thúc điều
            end_pos = dieu_matches[i + 1].start() if i + 1 < len(dieu_matches) else len(chuong.noi_dung)
            noi_dung = chuong.noi_dung[start_pos:end_pos].strip()
            dieu = Dieu(
                id=f"{chuong.id}_D{so_dieu}",
                chuong_id=chuong.id,
                van_ban_id=chuong.van_ban_id,
                so_dieu=so_dieu,
                ten_dieu=ten_dieu,
                noi_dung=noi_dung,
            )
            dieus.append(dieu)
        
        return dieus
    
    def _extract_id(self, text: str) -> str:
        """
        Trích xuất số hiệu văn bản pháp luật:
        - so_thu_tu
        - nam
        - loai_van_ban
        """
        pattern = re.compile(
            r'(?:Số|Luật\s*số|Nghị\s*định\s*số|Thông\s*tư\s*số|Quyết\s*định\s*số)'
            r'\s*:?\s*'
            r'([0-9]{1,4})'            # group 1: số thứ tự
            r'(?:/([0-9]{2,4}))?'      # group 2: năm (optional)
            r'/([A-ZĐ\-]+[0-9]*)',     # group 3: loại văn bản
            re.IGNORECASE
        )

        match = pattern.search(text)
        if not match:
            return None

        so_thu_tu, nam, loai_van_ban = match.groups()

        return f"{so_thu_tu}/{nam}/{loai_van_ban}" if nam else f"{so_thu_tu}/{loai_van_ban}"

    def split_full_document(self, text: str, ten_luat: str = '', van_ban_id: str = '') -> VanBan:
        """
        Tiền xử lý và chia nhỏ văn bản luật thành các chương và điều.
        Trả về đối tượng VanBan với cấu trúc chương và điều.
        """
        van_ban_id = (self._extract_id(text), van_ban_id)[van_ban_id != '']
        chuongs = self.split_into_chuongs(text, van_ban_id)
        
        for chuong in chuongs:
            dieus = self.split_into_dieus(chuong)
            chuong.dieus = dieus
        
        van_ban = VanBan(
            id=van_ban_id,
            ten=ten_luat,
            noi_dung=text,
            chuongs=chuongs
        )
        return van_ban

# Thêm tóm lược cho văn bản luật và chương
class LegalDocumentSummarizer:
    """Tóm lược văn bản luật"""
    
    def __init__(self, van_ban: VanBan, llm: OpenAI):
        self.van_ban = van_ban
        self.llm = llm
        
    
    def van_ban_summary(self, van_ban_summary_prompt: PromptTemplate = VANBAN_SUMMARIZATION_PROMPT) -> str:
        """Tóm lược văn bản luật"""
        # Lấy tên văn bản và tên các chương trong văn bản
        ten_chuongs = [chuong.ten_chuong for chuong in self.van_ban.chuongs]
        ten_van_ban = self.van_ban.ten
        # Lấy nội dung điều 1 của chương 1
        if self.van_ban.chuongs and self.van_ban.chuongs[0].dieus:
            dieu_1_chuong_1 = self.van_ban.chuongs[0].dieus[0].noi_dung
        else:
            dieu_1_chuong_1 = ""
        # Tạo tóm lược dựa trên thông tin trên
        try:
            response = self.llm.complete(van_ban_summary_prompt.format(ten_van_ban=ten_van_ban, ten_chuongs=', '.join(ten_chuongs), dieu_1_chuong_1=dieu_1_chuong_1))
        except Exception as e:
            print(f"Lỗi khi tóm tắt văn bản: {e}")
            return []
        return response.text.strip()
    
    def chuong_summary(self, chuong_summarization_prompt: PromptTemplate = CHUONG_SUMMARIZATION_PROMPT) -> List[str]:
        """Tóm lược chương trong văn bản luật"""
        chuongs_summary = []
        chuongs = self.van_ban.chuongs
        for chuong in chuongs:
            ten_chuong = chuong.ten_chuong
            ten_dieus = [dieu.ten_dieu for dieu in chuong.dieus]
            try:
                response = self.llm.complete(chuong_summarization_prompt.format(ten_chuong=ten_chuong, ten_dieus=', '.join(ten_dieus)))
            except Exception as e:
                print(f"Lỗi khi tóm tắt chương {chuong.so_chuong}: {e}")
                continue
            chuongs_summary.append(response.text.strip())
        return chuongs_summary
    
    def summarize(self):
        self.van_ban.tom_luoc = self.van_ban_summary()
        chuong_summaries = self.chuong_summary()
        for i, chuong in enumerate(self.van_ban.chuongs):
            chuong.tom_luoc = chuong_summaries[i]

# Tìm key-words trong văn bản , chương và điều luật 
class LegalDocumentKeywordExtractor:
    """Trích xuất từ khóa từ văn bản luật, chương và điều."""

    def _tokenize(self, text: str) -> str:
        text = text.lower()
        return word_tokenize(text)
    
    def _clean_tokens(self, tokens):
        cleaned = []
        for t in tokens:
            t = t.strip()
            # bỏ token chỉ là dấu câu
            if re.fullmatch(r'[^\wÀ-ỹ]+', t):
                continue
            cleaned.append(t)
        return cleaned
    
    def __init__(self, van_ban: VanBan):
        self.van_ban = van_ban
    
    # Tìm kiếm các từ khóa trong văn bản luật
    def find_van_ban_keywords(self) -> List[str]:
        document = ""
        document += self.van_ban.id + " " + self.van_ban.ten
        for chuong in self.van_ban.chuongs:
            document += " " + chuong.ten_chuong
        vanban_keywords = self._clean_tokens(self._tokenize(document))
        return vanban_keywords
    
    def find_chuong_keywords(self, chuong: Chuong) -> List[str]:
        document = ""
        document += chuong.ten_chuong
        for dieu in chuong.dieus:
            document += " " + dieu.so_dieu + " " + dieu.ten_dieu
        chuong_keywords = self._clean_tokens(self._tokenize(document))
        return chuong_keywords
    # Tìm kiếm từ khóa trong điều
    def find_dieu_keywords(self, dieu: Dieu) -> List[str]:
        document = ""
        document += dieu.ten_dieu
        document += " " + dieu.so_dieu
        dieu_keywords = self._clean_tokens(self._tokenize(document))
        return dieu_keywords
    
    def extract_all_keywords(self):
        """Trích xuất từ khóa cho văn bản luật, chương và điều."""
        # Từ khóa cho văn bản luật
        self.van_ban.key_words = self.find_van_ban_keywords()
        
        # Từ khóa cho các chương
        for chuong in self.van_ban.chuongs:
            chuong.key_words = self.find_chuong_keywords(chuong)
            # Từ khóa cho các điều trong chương
            for dieu in chuong.dieus:
                dieu.key_words = self.find_dieu_keywords(dieu)



class IndexingPipeline:
    def __init__(self):
        self.splitter = LegalDocumentSplitter()
        self.preprocessor = LegalDocumentPreprocessor()
        

    def test_process(self, llm = OpenAI):
        # Preprocess
        print("Xin chào, đây là chương trình giúp bạn tiền xử lí văn bản và lưu vào Document Store + Vector Store")
        file_path = input("Hãy nhập đường linh dẫn tới văn bản của bạn(pdf): ").strip()
        ten_luat = input("Hãy nhập tên luật: ").strip()
        so_hieu = input("Hãy nhập số hiệu văn bản (nếu ko có thì chương trình sẽ tự extract): ").strip()
        print("========")
        print("Đang chuyển đổi định dạng văn bản")
        text = self.preprocessor.preprocess_full_text(file_path, format="pdf")
        print("Đã chuyển đổi thành công")
        print("xxxxxxxx")
        print(f"{text[0:500]}")
        print("=========")
        # Splitting
        print("Đang phân tách văn bản của bạn")
        van_ban = self.splitter.split_full_document(text=text, ten_luat=ten_luat, van_ban_id=so_hieu)
        print("Đã thành công phân tách văn bản của bạn")
        print("xxxxxxxxxx")
        so_dieu = 0
        for chuong in van_ban.chuongs:
            so_dieu += len(chuong.dieus)
        print(f"Văn bản {van_ban.ten}, số hiệu {van_ban.id} có tổng cộng {len(van_ban.chuongs)} chương và {so_dieu} điều")
        print(f"Danh sách các chương bao gồm: ")
        for chuong in van_ban.chuongs:
            print(f"    Chương {chuong.so_chuong}: {chuong.ten_chuong}")
            for dieu in chuong.dieus:
                print(f"        Điều {dieu.so_dieu}: {dieu.ten_dieu}")
        print("=========")
        # Tóm lược văn bản
        print("Đang tóm lược văn bản")
        summarizer = LegalDocumentSummarizer(van_ban=van_ban, llm=llm)
        summarizer.summarize()
        print("Tóm tắt văn bản thành công")
        print("xxxxxxx")
        print(f"Tóm tắt văn bản: {van_ban.tom_luoc}")
        for chuong in van_ban.chuongs:
            print(f"    Chương {chuong.so_chuong}: {chuong.tom_luoc}")
        print("=========-")
        # Tìm từ khóa
        print("Đang tìm từ khóa văn bản")
        keyword_extractor = LegalDocumentKeywordExtractor(van_ban=van_ban)
        keyword_extractor.extract_all_keywords()
        print("Đã extract keyword thành công")
        print("xxxxxxxxx")
        print(f"Keyword văn bản: [{", ".join(van_ban.key_words)}]")
        for chuong in van_ban.chuongs:
            print(f"    Chương {chuong.so_chuong}: [{", ".join(chuong.key_words)}]")
            for dieu in chuong.dieus:
                print(f"        Điều {dieu.so_dieu}: [{", ".join(dieu.key_words)}]")


    def process(self, llm = OpenAI, file_path: str ="", ten_luat: str ="", id: str = "", format: str="pdf") -> VanBan:
        text = self.preprocessor.preprocess_full_text(file_path, format="pdf")
        van_ban = self.splitter.split_full_document(text=text, ten_luat=ten_luat, van_ban_id=id)
        summarizer = LegalDocumentSummarizer(van_ban=van_ban, llm=llm)
        summarizer.summarize()
        keyword_extractor = LegalDocumentKeywordExtractor(van_ban=van_ban)
        keyword_extractor.extract_all_keywords()
        return van_ban
    def save(self, van_ban: VanBan):
        mongo_repo = LegalMongoRepository()
        print("--------")
        print("Đang thêm văn bản vào Mongo")
        mongo_repo.insert_van_ban(van_ban_id=van_ban.id, ten=van_ban.ten, tom_luoc=van_ban.tom_luoc, keywords= van_ban.key_words)
        for chuong in van_ban.chuongs:
            mongo_repo.insert_chuong(chuong_id=chuong.id, van_ban_id=van_ban.id, so_chuong=chuong.so_chuong, ten=chuong.ten_chuong, tom_luoc=chuong.tom_luoc, keywords=chuong.key_words)
            for dieu in chuong.dieus:
                mongo_repo.insert_dieu(dieu_id=dieu.id, van_ban_id = van_ban.id, chuong_id=chuong.id, so_dieu=dieu.so_dieu, ten=dieu.ten_dieu, text=dieu.noi_dung, keywords=dieu.key_words)
        print("Thêm vào Mongo thành công")
        print("---------")
        print("Đang thêm vào Chroma")
        qdrant_repo = LegalQdrantRepository()
        embedder = BGEEmbedding(model_name="BAAI/bge-m3", device="cpu")
        van_ban_item = [
            {
                "id": str(uuid.uuid4()),
                "vector": embedder.embed(van_ban.tom_luoc),
                "payload": {
                    "type": "van_ban",
                    "van_ban_id": f"{van_ban.id}",
                    "chuong_id": "",
                    "dieu_id": ""
                }
            }
        ]
        qdrant_repo.upsert(items=van_ban_item)
        for chuong in van_ban.chuongs:
            chuong_item = [
                {
                    "id": str(uuid.uuid4()),
                    "vector": embedder.embed(chuong.tom_luoc),
                    "payload": {
                        "type": "chuong",
                        "van_ban_id": f"{van_ban.id}",
                        "chuong_id": f"{chuong.id}",
                        "dieu_id": ""
                    }
                }
            ]
            qdrant_repo.upsert(items=chuong_item)
            for dieu in chuong.dieus:
                dieu_item = [
                    {
                        "id": str(uuid.uuid4()),
                        "vector": embedder.embed(dieu.noi_dung),
                        "payload": {
                            "type": "dieu",
                            "van_ban_id": f"{van_ban.id}",
                            "chuong_id": f"{chuong.id}",
                            "dieu_id": f"{dieu.id}"
                        }
                    }
                ]
                qdrant_repo.upsert(items=dieu_item)
        print("Đã thêm vào Chroma thành công")

    def process_save(self, llm = OpenAI, file_path: str ="", ten_luat: str ="", id: str = "", format: str="pdf"):
        van_ban = self.process(llm = llm, file_path=file_path, ten_luat=ten_luat, id=id, format="pdf")
        self.save(van_ban=van_ban)

