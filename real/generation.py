from typing import List, Dict, Any

class LegalGenerator:
    def __init__(self, model_name: str = "gpt-4o", api_key: str = "YOUR_API_KEY"):
        # Bạn có thể thay bằng OpenAI, Anthropic hoặc các Model nội bộ (Llama-3)
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def _build_prompt(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        # Gom nội dung các điều luật lại thành một chuỗi context
        context_str = ""
        for i, doc in enumerate(contexts):
            if isinstance(doc, dict):
                context_str += f"[{i+1}] {doc.get('dieu_id')}: {doc.get('text')}\n\n"
            else:
                context_str += f"[{i+1}] {doc}\n\n"

        prompt = f"""
Bạn là một Trợ lý Luật sư ảo chuyên nghiệp tại Việt Nam. 
Nhiệm vụ của bạn là trả lời câu hỏi của người dùng một cách chính xác, khách quan dựa trên các điều luật được cung cấp dưới đây.

DƯỚI ĐÂY LÀ CÁC ĐIỀU LUẬT LIÊN QUAN:
---
{context_str}
---

CÂU HỎI CỦA NGƯỜI DÙNG:
"{query}"

YÊU CẦU TRẢ LỜI:
1. Nếu câu hỏi yêu cầu tra cứu đích danh (ví dụ: "Điều 4 là gì"), hãy trích dẫn chính xác nội dung điều đó.
2. Nếu câu hỏi là dạng tư vấn (ví dụ: "Tôi phải làm gì..."), hãy tổng hợp thông tin từ các điều luật để đưa ra lời khuyên.
3. Luôn luôn ghi rõ số Điều và tên văn bản luật (nếu có) trong câu trả lời.
4. Nếu thông tin trong các điều luật cung cấp không đủ để trả lời, hãy thành thật thông báo và yêu cầu người dùng cung cấp thêm chi tiết.
5. Ngôn ngữ: Trang trọng, chính xác, dễ hiểu.

CÂU TRẢ LỜI CỦA BẠN:
"""
        return prompt

    def generate_answer(self, query: str, reranked_docs: List[Dict[str, Any]]):
        if not reranked_docs:
            return "Xin lỗi, tôi không tìm thấy quy định pháp luật nào liên quan đến câu hỏi của bạn."

        full_prompt = self._build_prompt(query, reranked_docs)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "Bạn là trợ lý pháp luật chuyên nghiệp."},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.2, # Giữ nhiệt độ thấp để tránh AI "chế" luật
        )

        return response.choices[0].message.content
