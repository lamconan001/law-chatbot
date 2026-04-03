from embedding import BGEEmbedding
from storage_system import LegalQdrantRepository, LegalMongoRepository
from retrieval import Searcher, LegalReranker
from generation import LegalGenerator
import dotenv
dotenv.load_dotenv()
openai_api_key = dotenv.get_key(dotenv.find_dotenv(), "OPENAI_API_KEY")
query = 'Mua bán nhà đất cần những điều kiện gì?'
searcher = Searcher(query=query)
result = searcher.hybrid_search()
reranker = LegalReranker()
final_results = reranker.rerank(query=query, documents=result, top_k=5)

generator = LegalGenerator(model_name="gpt-4o-mini", api_key=openai_api_key)
answer = generator.generate_answer(query=query, reranked_docs=final_results)
print("Câu trả lời được tạo ra:")
print(answer)






