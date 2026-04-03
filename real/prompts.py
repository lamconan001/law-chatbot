# Các prompt dùng cho chương trình

from llama_index.core.prompts import PromptTemplate

DECOMPOSE_PROMPT = PromptTemplate(
    input_variables="question, max_subquestion",
    template=(
        "Hãy phân tách câu hỏi sau thành các câu hỏi nhỏ, rõ ràng và độc lập.\n"
        "Yêu cầu:\n"
        "1. Giữ nguyên ngữ cảnh pháp lý và không làm thay đổi ý nghĩa ban đầu của câu hỏi.\n"
        "2. Trả về danh sách các câu hỏi con, mỗi câu hỏi trên một dòng, bắt đầu bằng dấu gạch ngang (-).\n"
        "3. Không quá {max_subquestion} câu hỏi con. Nếu câu hỏi không cần phải phân tách thì không cần phân tách.\n"
        "Câu hỏi cần phân tích:\n"
        "{question}"
    ))
GENERALIZE_PROMPT = PromptTemplate(
    input_variables="sub_questions",
    template=(
        "Dựa trên các câu hỏi con sau, khái quát hóa thành một câu hỏi tổng quát hơn:\n" \
        "Yêu cầu:\n" \
        "1. Giữ nguyên ngữ cảnh pháp lý và không làm thay đổi ý nghĩa ban đầu của các câu hỏi con.\n" \
        "2. Trả về một câu hỏi tổng quát duy nhất, rõ ràng và súc tích.\n" \
        "3. Nếu chỉ có một câu hỏi con và không cần khái quát hóa, hãy trả về chính câu hỏi đó.\n" \
        "Các câu hỏi con:\n"
        "{sub_questions}"
    )
)

DOCUMENT_SEARCH_STRATEGY_PROMPT = PromptTemplate(
    input_variables="generalized_question",
    template=(
        "Dựa trên câu hỏi khái quát sau, xác định chiến lược tìm kiếm văn bản luật phù hợp.\n"
        "Yêu cầu:\n"
        "1. Chọn một trong hai chiến lược: 'semantic_search' hoặc 'keywords_search'.\n"
        "2. Trả về chỉ tên chiến lược đã chọn.\n"
        "Câu hỏi khái quát:\n"
        "{generalized_question}"
    ))

VANBAN_SUMMARIZATION_PROMPT = PromptTemplate(
    input_variables="ten_van_ban, ten_chuongs, dieu_1_chuong_1",
    template=(
        "Dựa trên tên văn bản, tên các chương và nội dung của Điều 1 Chương 1, hãy tạo một tóm lược ngắn gọn về văn bản luật này.\n"
        "Yêu cầu:\n"
        "1. Tóm lược nên ngắn gọn, súc tích và bao gồm các điểm chính của văn bản luật.\n"
        "2. Trả về tóm lược dưới dạng đoạn văn bản.\n"
        "3. Độ dài tóm tắt không vượt quá 500 ký tự.\n"
        "Tên văn bản: {ten_van_ban}\n"
        "Tên các chương: {ten_chuongs}\n"
        "Nội dung Điều 1 Chương 1: {dieu_1_chuong_1}"
    ))

DIEU_SUMMARIZATION_PROMPT = PromptTemplate(
    input_variables="so_dieu, ten_dieu, noi_dung",
    template=(
        "Hãy tóm tắt nội dung của điều luật sau:\n"
        "Yêu cầu:\n"
        "1. Tóm tắt nên ngắn gọn, súc tích và bao gồm các điểm chính của điều luật.\n"
        "2. Trả về tóm tắt dưới dạng đoạn văn bản.\n"
        "3. Độ dài tóm tắt không vượt quá 300 ký tự.\n"
        "Tên điều: {ten_dieu}\n"
        "Nội dung điều: {noi_dung}"
    ))

CHUONG_SUMMARIZATION_PROMPT = PromptTemplate(
    input_variables="ten_chuong, ten_dieus",
    template=(
        "Hãy tóm tắt nội dung của chương luật sau:\n"
        "Yêu cầu:\n"
        "1. Tóm tắt nên ngắn gọn, súc tích và bao gồm các điểm chính của chương luật.\n"
        "2. Trả về tóm tắt dưới dạng đoạn văn bản.\n"
        "3. Độ dài tóm tắt không vượt quá 400 ký tự.\n"
        "Tên chương: {ten_chuong}\n"
        "Tên các điều trong chương: {ten_dieus}"
    ))

ANSWER_GENERATION_PROMPT = PromptTemplate(
    input_variables="question, document_context",
    template=(
        "Dựa trên đoạn văn bản luật sau và câu hỏi của người dùng, hãy tạo một câu trả lời chính xác và đầy đủ.\n"
        "Yêu cầu:\n"
        "1. Câu trả lời phải dựa hoàn toàn vào thông tin trong đoạn văn bản luật được cung cấp.\n"
        "2. Trả về câu trả lời dưới dạng đoạn văn bản rõ ràng và dễ hiểu.\n"
        "3. Nếu thông tin trong đoạn văn bản luật không đủ để trả lời câu hỏi, hãy trả về 'Không đủ thông tin để trả lời câu hỏi.'\n"
        "Đoạn văn bản luật:\n"
        "{document_context}\n"
        "Câu hỏi của người dùng:\n"
        "{question}"
    ))

