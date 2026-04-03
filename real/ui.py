import streamlit as st
import os
from dotenv import load_dotenv
from retrieval import Searcher, LegalReranker, AdvancedRetriever, QuestionTransformer
from generation import LegalGenerator
from indexing_pipeline import IndexingPipeline
from llama_index.llms.openai import OpenAI
from storage_system import LegalMongoRepository
import tempfile
import gc

# Load biến môi trường
load_dotenv()

# Cấu hình trang
st.set_page_config(
    page_title="Chatbot Pháp Luật Việt Nam",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS tùy chỉnh
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-left: 5px solid #5a67d8;
    }
    .user-message strong {
        color: #fff;
    }
    .assistant-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border-left: 5px solid #e74c3c;
    }
    .assistant-message strong {
        color: #fff;
    }
    .sub-question-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #333;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #00d2ff;
        margin: 0.5rem 0;
        font-size: 0.95rem;
    }
    .sub-answer-box {
        background-color: #e8f5e9;
        color: #333;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #4caf50;
        margin: 0.5rem 0;
        font-size: 0.95rem;
    }
    .source-box {
        background-color: #fef5e7;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #f39c12;
        margin-top: 1rem;
        font-size: 0.9rem;
        color: #333;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #5a67d8 0%, #6c3f8f 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Hàm khởi tạo model với cache để tránh load lại nhiều lần
@st.cache_resource
def get_llm():
    """Khởi tạo LLM với cache"""
    api_key = os.getenv("OPENAI_API_KEY")
    return OpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0)

@st.cache_resource
def get_reranker():
    """Khởi tạo reranker với cache - chỉ load 1 lần"""
    return LegalReranker(device="cpu")

@st.cache_resource
def get_generator():
    """Khởi tạo generator với cache"""
    api_key = os.getenv("OPENAI_API_KEY")
    return LegalGenerator(model_name="gpt-4o-mini", api_key=api_key)

# Hàm dọn dẹp bộ nhớ
def clear_memory():
    """Dọn dẹp bộ nhớ"""
    gc.collect()

# Khởi tạo session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sử dụng cached resources
try:
    generator = get_generator()
    reranker = get_reranker()
    llm = get_llm()
except Exception as e:
    st.error(f"❌ Lỗi khởi tạo model: {str(e)}")
    st.stop()

# Header
st.markdown('<p class="main-header">⚖️ Trợ Lý Pháp Luật Việt Nam</p>', unsafe_allow_html=True)
st.markdown("---")

# Tạo tabs cho chatbot và upload
tab1, tab2 = st.tabs(["💬 Chat", "📤 Thêm Văn Bản"])

# Sidebar
with st.sidebar:
    st.header("⚙️ Cài Đặt")
    
    # Cài đặt threshold
    threshold = st.slider(
        "Ngưỡng độ tương đồng",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.05,
        help="Độ tương đồng tối thiểu giữa câu hỏi và văn bản pháp luật"
    )
    
    # Số lượng kết quả rerank
    top_k = st.slider(
        "Số lượng kết quả",
        min_value=1,
        max_value=10,
        value=3,
        help="Số lượng điều luật được hiển thị"
    )
    
    # Lựa chọn phương pháp tìm kiếm
    search_method = st.radio(
        "Phương pháp tìm kiếm",
        ["Hybrid Search", "Semantic Search"],
        help="Hybrid: kết hợp từ khóa và ngữ nghĩa. Semantic: chỉ dựa trên ngữ nghĩa"
    )
    
    # Tùy chọn phân tách câu hỏi
    use_decompose = st.checkbox(
        "Phân tách câu hỏi phức tạp",
        value=False,
        help="Tự động phân tách câu hỏi phức tạp thành các câu hỏi con để tìm kiếm chính xác hơn"
    )
    
    max_subquestions = st.slider(
        "Số câu hỏi con tối đa",
        min_value=1,
        max_value=3,
        value=2,
        help="Số lượng câu hỏi con tối đa khi phân tách"
    )
    
    st.markdown("---")
    
    # Nút xóa lịch sử
    if st.button("🗑️ Xóa Lịch Sử Chat"):
        st.session_state.messages = []
        clear_memory()
        st.rerun()
    
    st.markdown("---")
    
    # Thông tin
    st.markdown("""
    ### 📖 Hướng Dẫn Sử Dụng
    1. Nhập câu hỏi về pháp luật Việt Nam
    2. Hệ thống sẽ tìm kiếm các điều luật liên quan
    3. Nhận câu trả lời chi tiết kèm trích dẫn
    
    ### 💡 Ví Dụ Câu Hỏi
    - Điều 4 Bộ luật Dân sự quy định gì?
    - Quy định về hợp đồng mua bán nhà đất?
    - Thủ tục thành lập doanh nghiệp?
    """)

# TAB 1: CHAT
with tab1:
    # Hiển thị lịch sử chat (chỉ hiển thị 10 tin nhắn gần nhất)
    recent_messages = st.session_state.messages[-10:] if len(st.session_state.messages) > 10 else st.session_state.messages
    
    for message in recent_messages:
        if message["role"] == "user":
            st.markdown(
                f'<div class="chat-message user-message">'
                f'<strong>👤 Bạn:</strong><br>{message["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="chat-message assistant-message">'
                f'<strong>🤖 Trợ Lý:</strong><br>{message["content"]}</div>',
                unsafe_allow_html=True
            )
            
            # Hiển thị nguồn tham chiếu nếu có
            if "sources" in message and message["sources"]:
                with st.expander("📚 Xem Điều Luật Tham Khảo", expanded=False):
                    for idx, source in enumerate(message["sources"][:3], 1):
                        st.markdown(
                            f'<div class="source-box">'
                            f'<strong>Nguồn {idx}:</strong><br>{source}</div>',
                            unsafe_allow_html=True
                        )
        
            # Hiển thị câu hỏi con và câu trả lời nếu có
            if "sub_questions" in message and message["sub_questions"]:
                with st.expander("🔍 Xem Câu Hỏi Con & Câu Trả Lời Chi Tiết", expanded=False):
                    for i, sub_q in enumerate(message["sub_questions"], 1):
                        st.markdown(
                            f'<div class="sub-question-box">'  
                            f'<strong>❓ Câu hỏi {i}:</strong> {sub_q}</div>',
                            unsafe_allow_html=True
                        )
                        
                        # Hiển thị câu trả lời cho câu hỏi con
                        if "sub_answers" in message and i <= len(message["sub_answers"]):
                            st.markdown(
                                f'<div class="sub-answer-box">'  
                                f'<strong>✅ Trả lời {i}:</strong><br>{message["sub_answers"][i-1]}</div>',
                                unsafe_allow_html=True
                            )
    
    # Form nhập câu hỏi
    with st.form(key="question_form", clear_on_submit=True):
        col1, col2 = st.columns([6, 1])
        
        with col1:
            user_input = st.text_input(
                "Nhập câu hỏi của bạn:",
                placeholder="Ví dụ: Điều 4 Bộ luật Dân sự quy định về nguyên tắc gì?",
                label_visibility="collapsed"
            )
        
        with col2:
            submit_button = st.form_submit_button("Gửi 📤")

    # Xử lý khi người dùng gửi câu hỏi
    if submit_button and user_input:
        # Thêm câu hỏi vào lịch sử
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Hiển thị câu hỏi ngay lập tức
        st.markdown(
            f'<div class="chat-message user-message">'
            f'<strong>👤 Bạn:</strong><br>{user_input}</div>',
            unsafe_allow_html=True
        )
        
        # Hiển thị spinner trong khi xử lý
        with st.spinner("🔍 Đang tìm kiếm và phân tích..."):
            try:
                if use_decompose:
                    # Phân tách câu hỏi và xử lý
                    with st.status("⚙️ Đang phân tích câu hỏi...", expanded=True) as status:
                        st.write("📝 Phân tách câu hỏi thành các câu hỏi con...")
                        
                        # Lấy câu hỏi con
                        question_transformer = QuestionTransformer(
                            llm=llm,
                            max_subquestion=max_subquestions
                        )
                        sub_questions = question_transformer.decompose_question(user_input)
                        
                        if not sub_questions:
                            sub_questions = [user_input]
                        
                        st.write(f"✅ Đã phân tách thành {len(sub_questions)} câu hỏi")
                        for i, sq in enumerate(sub_questions, 1):
                            st.write(f"   {i}. {sq}")
                        
                        # Tìm kiếm và trả lời cho từng câu hỏi con
                        sub_answers = []
                        all_sources = []
                        
                        for i, sub_q in enumerate(sub_questions, 1):
                            st.write(f"\n🔍 Đang tìm kiếm cho câu hỏi {i}...")
                            
                            searcher = Searcher(query=sub_q)
                            if search_method == "Hybrid Search":
                                search_results = searcher.hybrid_search(threshold=threshold)
                            else:
                                search_results = searcher.semantic_search(threshold=threshold)
                            
                            # Rerank
                            reranked_docs = reranker.rerank(
                                query=sub_q,
                                documents=search_results,
                                top_k=top_k
                            )
                            
                            # Generate câu trả lời cho câu hỏi con
                            sub_answer = generator.generate_answer(
                                query=sub_q,
                                reranked_docs=reranked_docs
                            )
                            sub_answers.append(sub_answer)
                            
                            # Lưu nguồn
                            if isinstance(search_results, dict):
                                all_sources.extend(list(search_results.values())[:2])
                            
                            # Dọn dẹp bộ nhớ sau mỗi câu hỏi con
                            clear_memory()
                        
                        # Tạo câu trả lời tổng hợp
                        st.write("\n📝 Đang tổng hợp câu trả lời...")
                        
                        combined_context = "\n\n".join([
                            f"Câu hỏi {i}: {sq}\nTrả lời: {ans}"
                            for i, (sq, ans) in enumerate(zip(sub_questions, sub_answers), 1)
                        ])
                        
                        final_prompt = f"""
                        Dựa trên các câu trả lời chi tiết cho từng câu hỏi con dưới đây, hãy tổng hợp thành một câu trả lời hoàn chỉnh và mạch lạc cho câu hỏi gốc.
                        
                        CÂU HỎI GỐC: {user_input}
                        
                        CÁC CÂU TRẢ LỜI CHI TIẾT:
                        {combined_context}
                        
                        YÊU CẦU:
                        - Tổng hợp thông tin từ tất cả các câu trả lời
                        - Trình bày mạch lạc, dễ hiểu
                        - Giữ nguyên các trích dẫn điều luật quan trọng
                        - Không thêm thông tin không có trong các câu trả lời trên
                        
                        CÂU TRẢ LỜI TỔNG HỢP:
                        """
                        
                        final_answer = llm.complete(final_prompt).text
                        
                        status.update(label="✅ Hoàn thành!", state="complete")
                    
                    # Thêm vào lịch sử
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": final_answer,
                        "sources": list(set(all_sources))[:5],  # Chỉ lưu 5 nguồn
                        "sub_questions": sub_questions,
                        "sub_answers": sub_answers
                    })
                
                else:
                    # Không phân tách - xử lý trực tiếp
                    searcher = Searcher(query=user_input)
                    
                    if search_method == "Hybrid Search":
                        search_results = searcher.hybrid_search(threshold=threshold)
                    else:
                        # Semantic search trả về list points, cần chuyển sang dict
                        points = searcher.semantic_search(threshold=threshold)
                        
                        # Chuyển đổi points sang dict format
                        search_results = {}
                        mongo_repo = LegalMongoRepository()
                        
                        for point in points:
                            dieu_id = point.payload.get('dieu_id')
                            van_ban_id = point.payload.get('van_ban_id')
                            
                            # Lấy thông tin chi tiết từ MongoDB
                            van_ban = mongo_repo.get_van_ban_by_id(van_ban_id=van_ban_id)
                            dieu_info = mongo_repo.get_dieu_by_ids(dieu_ids=[dieu_id])
                            
                            if dieu_info:
                                dieu = dieu_info[0]
                                ten_van_ban = van_ban.get('ten') if van_ban else ''
                                search_results[dieu_id] = f"{ten_van_ban} - Điều {dieu.get('so_dieu')}. {dieu.get('ten')}\n{dieu.get('text')}"
                    
                    # Rerank
                    reranked_docs = reranker.rerank(
                        query=user_input,
                        documents=search_results,
                        top_k=top_k
                    )
                    
                    # Generate câu trả lời
                    answer = generator.generate_answer(
                        query=user_input,
                        reranked_docs=reranked_docs
                    )
                    
                    # Lưu nguồn tham chiếu
                    sources = []
                    if isinstance(search_results, dict):
                        sources = list(search_results.values())[:top_k]
                    
                    # Thêm câu trả lời vào lịch sử
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                
                # Dọn dẹp bộ nhớ sau khi hoàn thành
                clear_memory()
                
                # Rerun để hiển thị câu trả lời
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Đã xảy ra lỗi: {str(e)}")
                st.error("Vui lòng thử lại với câu hỏi ngắn gọn hơn hoặc tắt 'Phân tách câu hỏi phức tạp'")
                clear_memory()

# TAB 2: UPLOAD VÀ INDEXING
with tab2:
    st.subheader("📤 Thêm Văn Bản Pháp Luật Mới")
    st.markdown("Tải lên file PDF văn bản pháp luật để thêm vào cơ sở dữ liệu")
    
    # Form upload file
    with st.form(key="upload_form", clear_on_submit=True):
        # Upload file
        uploaded_file = st.file_uploader(
            "Chọn file PDF văn bản pháp luật",
            type=["pdf"],
            help="Hỗ trợ định dạng PDF"
        )
        
        # Thông tin văn bản
        col1, col2 = st.columns(2)
        
        with col1:
            ten_van_ban = st.text_input(
                "Tên văn bản *",
                placeholder="Ví dụ: Bộ luật Dân sự 2015",
                help="Nhập tên đầy đủ của văn bản pháp luật"
            )
        
        with col2:
            so_hieu = st.text_input(
                "Số hiệu văn bản",
                placeholder="Ví dụ: 91/2015/QH13",
                help="Để trống nếu muốn hệ thống tự động trích xuất"
            )
        
        # Nút submit
        submit_upload = st.form_submit_button("🚀 Bắt Đầu Indexing")
    
    # Xử lý upload và indexing
    if submit_upload:
        if not uploaded_file:
            st.error("⚠️ Vui lòng chọn file PDF để tải lên!")
        elif not ten_van_ban:
            st.error("⚠️ Vui lòng nhập tên văn bản!")
        else:
            # Tạo file tạm thời
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name
            
            try:
                # Hiển thị progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Bước 1: Khởi tạo pipeline
                status_text.text("🔧 Đang khởi tạo pipeline...")
                progress_bar.progress(10)
                
                api_key = os.getenv("OPENAI_API_KEY")
                llm = OpenAI(model="gpt-4o-mini", api_key=api_key)
                pipeline = IndexingPipeline()
                
                # Bước 2: Xử lý văn bản
                status_text.text("📄 Đang đọc và phân tích văn bản...")
                progress_bar.progress(30)
                
                van_ban = pipeline.process(
                    llm=llm,
                    file_path=tmp_file_path,
                    ten_luat=ten_van_ban,
                    id=so_hieu if so_hieu else "",
                    format="pdf"
                )
                
                # Bước 3: Lưu vào database
                status_text.text("💾 Đang lưu vào cơ sở dữ liệu...")
                progress_bar.progress(70)
                
                pipeline.save(van_ban=van_ban)
                
                # Hoàn thành
                progress_bar.progress(100)
                status_text.text("✅ Hoàn thành!")
                
                # Hiển thị thông tin văn bản
                st.success(f"🎉 Đã thêm thành công văn bản: **{van_ban.ten}**")
                
                # Thống kê
                so_chuong = len(van_ban.chuongs)
                so_dieu = sum(len(chuong.dieus) for chuong in van_ban.chuongs)
                
                st.info(f"""
                **Thông tin văn bản:**
                - 📋 Số hiệu: {van_ban.id}
                - 📚 Số chương: {so_chuong}
                - 📜 Số điều: {so_dieu}
                """)
                
                # Hiển thị chi tiết
                with st.expander("🔍 Xem Chi Tiết Cấu Trúc"):
                    for chuong in van_ban.chuongs[:5]:
                        st.markdown(f"**Chương {chuong.so_chuong}: {chuong.ten_chuong}**")
                        for dieu in chuong.dieus[:5]:
                            st.text(f"  • Điều {dieu.so_dieu}: {dieu.ten_dieu}")
                
                # Dọn dẹp bộ nhớ
                clear_memory()
                
            except Exception as e:
                st.error(f"❌ Đã xảy ra lỗi trong quá trình indexing: {str(e)}")
                st.error("Vui lòng kiểm tra file PDF và thử lại.")
            
            finally:
                # Xóa file tạm
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)

# Footer
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: gray; font-size: 0.9rem;">'
    'Chatbot Pháp Luật Việt Nam © 2026 | Phát triển với ❤️ bởi lamconan'
    '</p>',
    unsafe_allow_html=True
)