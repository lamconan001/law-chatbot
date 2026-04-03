# Phân tích câu hỏi thành các câu hỏi con và thực hiện khái quát hóa câu hỏi
# Ví dụ: 
# Câu hỏi gốc: 
# "Tôi đang tìm hiểu và chuẩn bị tham gia giao dịch mua bất động sản. 
# Tôi đề nghị cơ quan chức năng hướng dẫn xử lý tình huống về ký hợp đồng đặt cọc/mua bán bất động sản hình thành trong tương lai để tôi hiểu và tuân thủ đúng quy định: 
# Một văn bản thỏa thuận ký giữa khách hàng và công ty môi giới bất động sản quy định: khách hàng đặt cọc đủ 100% giá trị sản phẩm theo lộ trình (10% vốn tự có khi ký; 
# ngân hàng giải ngân 80% khoản vay sau 1-1,5 tháng; thêm 10% vốn tự có sau khoảng 2 tháng), trong khi dự án chưa đủ điều kiện mở bán. 
# Văn bản thỏa thuận ràng buộc khách hàng phải ký hợp đồng mua bán, nếu không ký thì bị phạt cọc và thu phí tư vấn 0,5% giá trị bất động sản, nếu ký thì không thu phí này. 
# Khi ký hợp đồng mua bán, đồng thời ký Phụ lục 01 văn bản thỏa thuận, trong đó: 
# 20% vốn tự có chuyển thành hợp đồng vay cho chủ đầu tư; 10% do môi giới giữ lại (từ khoản ngân hàng đã giải ngân); 
# 70% còn lại chuyển qua "tài khoản chỉ định" thực chất là tài khoản thu nợ của ngân hàng để thanh toán giá bán sản phẩm theo hợp đồng mua bán. 
# Tôi xin hỏi, văn bản thỏa thuận như trên có phải là thỏa thuận dân sự hợp pháp, hay thực chất là hình thức trá hình của hợp đồng đặt cọc/mua bán bất động sản và 
# huy động vốn trái luật khi dự án chưa đủ điều kiện mở bán? 
# Văn bản thỏa thuận và Phụ lục 01 có thuộc trường hợp giao dịch vô hiệu theo Điều 123, 124 Bộ luật Dân sự 2015 hay không?""

# Thì sau khi phân tích, ta có thể tách thành các câu hỏi con như sau:
# Câu hỏi 1: Văn bản thỏa thuận ký giữa khách hàng và công ty môi giới bất động sản quy định về việc đặt cọc 100% giá trị sản phẩm khi dự án chưa đủ điều kiện mở bán có phải là thỏa thuận dân sự hợp pháp không?
# Câu hỏi 2: Việc ràng buộc khách hàng phải ký hợp đồng mua bán và phạt cọc nếu không ký có hợp pháp không?
# Câu hỏi 3: Việc chuyển đổi 20% vốn tự có thành hợp đồng vay cho chủ đầu tư và sử dụng "tài khoản chỉ định" để thanh toán giá bán sản phẩm có vi phạm quy định về huy động vốn không?

# Cuối cùng, ta khái quát hóa câu hỏi thành:
# "Quy định pháp luật về việc đặt cọc, ký hợp đồng mua bán bất
# động sản hình thành trong tương lai khi dự án chưa đủ điều kiện mở bán và các hình thức huy động vốn liên quan."
# # Sau khi phân tích và khái quát hóa, ta sẽ sử dụng các câu hỏi con và câu hỏi khái quát để tìm kiếm thông tin trong văn bản pháp luật và trả lời người dùng.

# Phân tích câu hỏi thành các câu hỏi con
# Dùng LLM để phân tách câu hỏi phức tạp thành các câu hỏi con rõ ràng hơn
# Sử dụng thư viện LalamaIndex và OpenAI
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.openai import OpenAI
from typing import List, Dict, Any
from prompts import DECOMPOSE_PROMPT, GENERALIZE_PROMPT
from storage_system import LegalMongoRepository, LegalQdrantRepository
from embedding import BGEEmbedding
import os 


class Searcher:
    # Lớp để tìm kiếm
    def __init__(self, query: str):
        self.vector_db = LegalQdrantRepository()
        self.doc_db = LegalMongoRepository
        self.query = query
        self.embeder = BGEEmbedding()
        
    def semantic_search(self, threshold: float = 0.6): # threshold thường là float (0.0 - 1.0)
        vector_query = self.embeder.embed(self.query)
        
        # 1. Tìm Văn bản liên quan nhất
        van_ban_points = self.vector_db.van_ban_semantic_search(vector=vector_query, limit=3)
        van_ban_ids = [point.payload.get('van_ban_id') for point in van_ban_points]
        
        # Dùng set để tránh trùng lặp Điều luật
        final_dieu_points = {} # Dùng dict {id: point} để dễ quản lý

        for van_ban_id in van_ban_ids:
            # 2. Tìm Chương trong từng Văn bản
            chuong_points = self.vector_db.chuong_semantic_search(vector=vector_query, van_ban_id=van_ban_id, limit=3)
            
            for chuong_point in chuong_points:
                chuong_id = chuong_point.payload.get('chuong_id')
                
                # 3. Tìm Điều trong từng Chương
                dieu_points = self.vector_db.dieu_semantic_search(
                    vector=vector_query, 
                    van_ban_id=van_ban_id, 
                    chuong_id=chuong_id, 
                    limit=10
                )
                for dp in dieu_points:
                    if dp.score >= threshold:
                        final_dieu_points[dp.payload.get('dieu_id')] = dp

            # 4. Tìm kiếm "Bảo hiểm" (Global search trong văn bản)
            extra_dieu_points = self.vector_db.dieu_semantic_search(
                vector=vector_query, 
                van_ban_id=van_ban_id, 
                limit=5
            )
            for edp in extra_dieu_points:
                if edp.score >= threshold:
                    final_dieu_points[edp.payload.get('dieu_id')] = edp

        # Chuyển từ dict về list các point để đưa vào Reranker hoặc LLM
        return list(final_dieu_points.values())
    
    def hybrid_search(self, threshold: float = 0.6):
        vector_query = self.embeder.embed(self.query)
        
        # 1. Tìm Văn bản liên quan nhất
        van_ban_points = self.vector_db.van_ban_semantic_search(vector=vector_query, limit=3)
        van_ban_ids = [point.payload.get('van_ban_id') for point in van_ban_points]

        # Tìm kiếm điều luật trong văn bản dựa trên từ khóa
        mongo_repo = LegalMongoRepository()
        dieu_results = {}
        # Dùng set để tránh trùng lặp Điều luật

        for van_ban_id in van_ban_ids:
            van_ban = mongo_repo.get_van_ban_by_id(van_ban_id=van_ban_id)
            ten_van_ban = van_ban.get('ten')
            keyword_dieu_results = mongo_repo.dieu_keyword_search(query=self.query, van_ban_id=van_ban_id, limit=5)
            for kd in keyword_dieu_results:
                dieu_id = kd.get('_id')
                noi_dung = kd.get('text')
                ten = kd.get('ten')
                so_dieu = kd.get('so_dieu')
                dieu_results[dieu_id] = f"{ten_van_ban} - Điều {so_dieu}. {ten}\n{noi_dung}"
            # 2. Tìm Chương trong từng Văn bản
            ids = []
            chuong_points = self.vector_db.chuong_semantic_search(vector=vector_query, van_ban_id=van_ban_id, limit=3)
            
            for chuong_point in chuong_points:
                chuong_id = chuong_point.payload.get('chuong_id')
                
                # 3. Tìm Điều trong từng Chương
                dieu_points = self.vector_db.dieu_semantic_search(
                    vector=vector_query, 
                    van_ban_id=van_ban_id, 
                    chuong_id=chuong_id, 
                    limit=10
                )
                for dp in dieu_points:
                    if dp.score >= threshold:
                        ids.append(dp.payload.get('dieu_id'))

            # 4. Tìm kiếm "Bảo hiểm" (Global search trong văn bản)
            extra_dieu_points = self.vector_db.dieu_semantic_search(
                vector=vector_query, 
                van_ban_id=van_ban_id, 
                limit=5
            )
            for edp in extra_dieu_points:
                if edp.score >= threshold:
                    ids.append(edp.payload.get('dieu_id'))

        # Tìm nôi dụng điều luật từ ids
            semantic_results = mongo_repo.get_dieu_by_ids(dieu_ids=ids)
            for sd in semantic_results:
                dieu_id = sd.get('_id')
                noi_dung = sd.get('text')
                ten = sd.get('ten')
                so_dieu = sd.get('so_dieu')
                dieu_results[dieu_id] = f"{ten_van_ban} - Điều {so_dieu}. {ten}\n{noi_dung}"
        return dieu_results

from FlagEmbedding import FlagReranker
from typing import List, Dict, Any

class LegalReranker:
    def __init__(self, model_name: str = 'BAAI/bge-reranker-v2-m3', device: str = "cpu"):
        """
        model_name: 'BAAI/bge-reranker-v2-m3' (Khuyên dùng cho Tiếng Việt)
        device: 'cuda' cho GPU, 'cpu' cho CPU. Nếu None sẽ tự động chọn.
        """
        self.reranker = FlagReranker(model_name, use_fp16=True, device=device) 

    def rerank(self, query: str, documents: Dict[str, str], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        query: Câu hỏi người dùng
        documents: Dict các document từ MongoDB (chứa 'ten', 'text', 'so_dieu')
        """
        if not documents:
            return []

        # 1. Chuẩn bị dữ liệu đầu vào: [[query, passage1], [query, passage2], ...]
        # Mẹo: Kết hợp 'ten' và 'text' để Reranker có ngữ cảnh tốt nhất
        pairs = []
        doc_list = list(documents.values())
        for doc in doc_list:
            pairs.append([query, f"{doc}"])

        # 2. Chấm điểm (Sử dụng compute_score cho FlagEmbedding)
        # Điểm số trả về thường là điểm thô (logits), càng cao càng liên quan
        scores = self.reranker.compute_score(pairs)

        # 3. Gán điểm lại cho tài liệu
        # Lưu ý: scores có thể là list hoặc float tùy số lượng pairs
        if isinstance(scores, float):
            scores = [scores]
    
        for doc, score in zip(doc_list, scores):
            if isinstance(doc, dict):
                doc['rerank_score'] = score
            else:
                doc = {'content': doc, 'rerank_score': score}

        # 4. Sắp xếp và lấy Top K
        reranked_docs = sorted(doc_list, key=lambda x: x.get('rerank_score', 0) if isinstance(x, dict) else 0, reverse=True)

        return reranked_docs[:top_k]


class QuestionTransformer:
    """Lớp để phân tách và khái quát hóa câu hỏi pháp lý."""
    
    def __init__(self, llm: OpenAI, max_subquestion: int = 3, decompose_prompt: PromptTemplate = DECOMPOSE_PROMPT):
        self.llm = llm
        self.max_subquestion = max_subquestion
        self.decompose_promt = decompose_prompt

        
        
    def decompose_question(self, question: str) -> List[str]:
        """
        Phân tách một câu hỏi phức tạp thành các câu hỏi nhỏ hơn.
        """
        try:
            response = self.llm.complete(self.decompose_promt.format(question=question, max_subquestion=self.max_subquestion))
        except Exception as e:
            print(f"Lỗi khi phân tách câu hỏi: {e}")
            return []
        sub_questions = [
            line.strip("- ").strip()
            for line in response.text.split("\n")
            if line.strip()
        ]
        return sub_questions


class AdvancedRetriever:
    """
    Lớp kết hợp đầy đủ các bước:
    1. Phân tích câu hỏi thành câu hỏi con
    2. Thực hiện hybrid search trên các câu hỏi con
    3. Kết hợp và loại bỏ trùng lặp kết quả
    4. Rerank để lấy kết quả tốt nhất
    """
    
    def __init__(self, llm: OpenAI, reranker: LegalReranker = None, max_subquestion: int = 3):
        """
        Args:
            llm: OpenAI LLM để phân tách câu hỏi
            reranker: LegalReranker để rerank kết quả (optional)
            max_subquestion: Số câu hỏi con tối đa
        """
        self.llm = llm
        self.question_transformer = QuestionTransformer(llm=llm, max_subquestion=max_subquestion)
        self.reranker = reranker
        self.embedder = BGEEmbedding()
        self.vector_db = LegalQdrantRepository()
        self.mongo_repo = LegalMongoRepository()
    
    def retrieve(self, question: str, threshold: float = 0.6, top_k: int = 5) -> Dict[str, str]:
        """
        Thực hiện toàn bộ quá trình retrieval:
        1. Phân tách câu hỏi
        2. Hybrid search cho từng câu hỏi con
        3. Kết hợp kết quả
        4. Rerank (nếu có)
        
        Args:
            question: Câu hỏi gốc của người dùng
            threshold: Ngưỡng độ tương đồng
            top_k: Số lượng kết quả cuối cùng
            
        Returns:
            Dict[dieu_id, formatted_content]
        """
        # Bước 1: Phân tách câu hỏi thành các câu hỏi con
        print(f"📝 Câu hỏi gốc: {question}")
        sub_questions = self.question_transformer.decompose_question(question)
        
        if not sub_questions:
            print("⚠️ Không phân tách được câu hỏi, sử dụng câu hỏi gốc")
            sub_questions = [question]
        else:
            print(f"✅ Đã phân tách thành {len(sub_questions)} câu hỏi con:")
            for i, sq in enumerate(sub_questions, 1):
                print(f"   {i}. {sq}")
        
        # Bước 2: Thực hiện hybrid search cho từng câu hỏi con
        all_results = {}
        
        for i, sub_q in enumerate(sub_questions, 1):
            print(f"🔍 Đang tìm kiếm cho câu hỏi {i}...")
            searcher = Searcher(query=sub_q)
            results = searcher.hybrid_search(threshold=threshold)
            
            # Gộp kết quả (tự động loại trùng vì dùng dict)
            all_results.update(results)
        
        print(f"📚 Tổng cộng tìm thấy {len(all_results)} điều luật duy nhất")
        
        # Bước 3: Rerank nếu có reranker
        if self.reranker and len(all_results) > top_k:
            print(f"🔄 Đang rerank để lấy top {top_k} kết quả...")
            reranked = self.reranker.rerank(
                query=question,  # Dùng câu hỏi gốc để rerank
                documents=all_results,
                top_k=top_k
            )
            
            # Chuyển về dict format
            final_results = {}
            for doc in reranked:
                # Tìm key tương ứng trong all_results
                for key, value in all_results.items():
                    if value == doc or (isinstance(doc, dict) and doc.get('content') == value):
                        final_results[key] = value
                        break
            
            return final_results
        else:
            # Không rerank, chỉ lấy top_k đầu tiên
            return dict(list(all_results.items())[:top_k])
    
    def retrieve_simple(self, question: str, threshold: float = 0.6, top_k: int = 5) -> Dict[str, str]:
        """
        Phiên bản đơn giản: chỉ search câu hỏi gốc (không phân tách)
        """
        print(f"🔍 Đang tìm kiếm cho: {question}")
        searcher = Searcher(query=question)
        results = searcher.hybrid_search(threshold=threshold)
        
        if self.reranker and len(results) > top_k:
            print(f"🔄 Đang rerank để lấy top {top_k} kết quả...")
            reranked = self.reranker.rerank(
                query=question,
                documents=results,
                top_k=top_k
            )
            
            final_results = {}
            for doc in reranked:
                for key, value in results.items():
                    if value == doc or (isinstance(doc, dict) and doc.get('content') == value):
                        final_results[key] = value
                        break
            
            return final_results
        else:
            return dict(list(results.items())[:top_k])


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    # Test AdvancedRetriever
    api_key = os.getenv("OPENAI_API_KEY")
    llm = OpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
    reranker = LegalReranker(device="cpu")
    
    retriever = AdvancedRetriever(llm=llm, reranker=reranker, max_subquestion=3)
    
    user_question = input("Nhập câu hỏi của bạn (hoặc 'exit' để thoát): ")
    
    while user_question.strip().lower() != "exit":
        print("\n" + "="*80)
        results = retriever.retrieve(question=user_question, threshold=0.6, top_k=5)
        
        print("\n📋 KẾT QUẢ CUỐI CÙNG:")
        print("="*80)
        for i, (dieu_id, content) in enumerate(results.items(), 1):
            print(f"\n[{i}] {dieu_id}")
            print(f"{content[:300]}..." if len(content) > 300 else content)
        
        print("\n" + "="*80)
        user_question = input("\nNhập câu hỏi của bạn (hoặc 'exit' để thoát): ")
    
