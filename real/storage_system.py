from typing import List, Dict, Optional, Any
from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    Distance, 
    VectorParams
)

from pymongo import MongoClient
import re

class LegalMongoRepository:
    
    def __init__(self, mongo_uri: str = "mongodb://localhost:27017", db_name: str = "legal_db"):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.van_ban_col = self.db["van_ban"]
        self.chuong_col = self.db["chuong"]
        self.dieu_col = self.db["dieu"]
        self._ensure_indexes()

    # Index
    def _ensure_indexes(self):
        # Dieu
        self.dieu_col.create_index("van_ban_id")
        self.dieu_col.create_index("chuong_id")
        self.dieu_col.create_index("so_dieu")
        # Index tìm kiếm nhanh theo mảng (Multikey Index)
        self.dieu_col.create_index("keywords")
        
        # Full-text Index: Tập trung toàn lực vào keywords
        self.dieu_col.create_index(
            [("keywords", "text"), ("text", "text")],
            weights={"keywords": 20, "text": 1}, # Tăng vọt trọng số cho keywords
            name="dieu_keyword_heavy_search"
        )
    
    def insert_van_ban(
            self,
            van_ban_id: str,
            ten: str,
            tom_luoc: str,
            keywords: Optional[List[str]] = None
    ):
        doc = {
            "_id": van_ban_id,
            "ten": ten,
            "tom_luoc": tom_luoc,
            "keywords": keywords or []
        }
        self.van_ban_col.replace_one({"_id": doc["_id"]}, doc, upsert=True)
    
    # Chuong
    def insert_chuong(
            self,
            chuong_id: str,
            van_ban_id: str,
            so_chuong: str,
            ten: str,
            tom_luoc: str,
            keywords: Optional[List[str]]= None
    ):
        doc = {
            "_id": chuong_id,
            "van_ban_id": van_ban_id,
            "so_chuong": so_chuong,
            "ten": ten,
            "tom_luoc": tom_luoc,
            "keywords": keywords or []
        }
        self.chuong_col.replace_one({"_id": doc["_id"]}, doc, upsert=True)

    # Dieu
    def insert_dieu(
        self,
        dieu_id: str,
        van_ban_id: str,
        chuong_id: str,
        so_dieu: str,
        ten: str,
        text: str,
        keywords: Optional[List[str]] = None
    ):
        doc = {
            "_id": dieu_id,
            "van_ban_id": van_ban_id,
            "chuong_id": chuong_id,
            "so_dieu": so_dieu,
            "ten": ten,
            "text": text,       
            "keywords": keywords or []
        }
        self.dieu_col.replace_one({"_id": doc["_id"]}, doc, upsert=True)


    def get_dieu_by_van_ban(self, van_ban_id: str):
        return List(self.dieu_col.find({"van_ban_id": van_ban_id}))

    def get_dieu_by_chuong(self, chuong_id: str):
        return List(self.dieu_col.find({"chuong_id": chuong_id}))
    
    def get_chuong_by_van_ban(self, van_ban_id: str):
        return List(self.chuong_col.fin({"van_ban_id": van_ban_id}))
    
    def get_dieu_by_ids(self, dieu_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch Dieu documents theo đúng thứ tự id truyền vào
        """
        if not dieu_ids:
            return []
        return list(
            self.dieu_col.find({"_id": {"$in": dieu_ids}})
        )

    def get_van_ban_by_ids(self, van_ban_ids: List[str]) -> List[Dict[str, Any]]:
        if not van_ban_ids:
            return []

        return list(
            self.van_ban_col.find({"_id": {"$in": van_ban_ids}})
        )
    def get_van_ban_by_id(self, van_ban_id: str) -> Optional[Dict[str, Any]]:
        return self.van_ban_col.find_one({"_id": van_ban_id})
    
    def get_chuong_by_ids(self, chuong_ids: List[str]) -> List[Dict[str, Any]]:
        if not chuong_ids:
            return []

        return list(
            self.chuong_col.find({"_id": {"$in": chuong_ids}})
        )

    def dieu_keyword_search(self, query: str, van_ban_id: Optional[str] = None, limit: int = 20):
        if not query: return []
        # Trích xuất số điều
        match = re.search(r'(?:điều|đ)\s*(\d+)', query, re.IGNORECASE)
        
        exact_results = []
        if match:
            so_dieu_str = match.group(1)
            exact_filter = {"so_dieu": so_dieu_str}
            if van_ban_id:
                exact_filter["van_ban_id"] = van_ban_id
                
            # Lấy kết quả chính xác
            cursor = self.dieu_col.find(exact_filter).limit(5)
            for doc in cursor:
                # Gán score giả cực cao để không bị lỗi Key và giữ vị trí đầu
                doc['score'] = 100.0 
                exact_results.append(doc)

        # 2. Full-text search cho phần còn lại
        exclude_ids = [doc["_id"] for doc in exact_results]
        text_filter = {"$text": {"$search": query}, "_id": {"$nin": exclude_ids}}
        if van_ban_id:
            text_filter["van_ban_id"] = van_ban_id
            
        text_results = list(self.dieu_col.find(
            text_filter,
            {"score": {"$meta": "textScore"}}
        ).sort([("score", {"$meta": "textScore"})]).limit(limit - len(exact_results)))

        return exact_results + text_results
    
        
class LegalQdrantRepository:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "legal_vectors"
    ):
        self.client = QdrantClient(host=host, port=port)
        self.collection = collection_name
        self._ensure_collection()

    def _ensure_collection(self):
        """Create collection if it doesn't exist"""
        try:
            self.client.get_collection(self.collection)
        except:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    "text_vector": VectorParams(size=1024, distance=Distance.COSINE)
                    }
            )

    def upsert(
        self,
        items: List[Dict[str, Any]]
    ):
        """
        items = [
            {
                "id": "",
                "vector": [...],
                "payload": {
                    "type": "van_ban | chuong | dieu",
                    "van_ban_id": "",
                    "chuong_id": "",
                    "dieu_id": ""
                }
            }
        ]
        """

        points = [
            PointStruct(
                id=item["id"],
                vector={
                    "text_vector" : item["vector"] 
                },
                payload=item["payload"]
            )
            for item in items
        ]

        self.client.upsert(
            collection_name=self.collection,
            points=points
        )

    
    def van_ban_semantic_search(self, vector, limit: int = 10):
        """Search for van_ban documents similar to the vector"""
        results = self.client.query_points(
            collection_name=self.collection,
            query=vector,           # Truyền list [0.1, 0.2...] trực tiếp ở đây
            using="text_vector",    # 🔑 Khai báo tên vector ở đây
            query_filter=models.Filter(
                must=[models.FieldCondition(key="type", match=models.MatchValue(value="van_ban"))]
            ),
            limit=limit,
            with_payload=True
        )
        return results.points
    
    def chuong_semantic_search(self, vector, van_ban_id: Optional[str] = None, limit: int = 10):
        """Search for chuong documents similar to the vector"""
        
        # 1. Xây dựng danh sách các điều kiện lọc (Filters)
        # Lưu ý: Không dùng "payload.type", chỉ dùng "type"
        must_conditions = [
            models.FieldCondition(key="type", match=models.MatchValue(value="chuong"))
        ]
        
        # Nếu có van_ban_id thì thêm vào điều kiện lọc
        if van_ban_id:
            must_conditions.append(
                models.FieldCondition(key="van_ban_id", match=models.MatchValue(value=van_ban_id))
            )

        # 2. Thực hiện query_points
        results = self.client.query_points(
            collection_name=self.collection,
            query=vector,           # Vector câu hỏi (list số)
            using="text_vector",    # 🔑 Đảm bảo trùng với tên lúc tạo collection
            query_filter=models.Filter(must=must_conditions),
            limit=limit,
            with_payload=True       # Để lấy về nội dung chương
        )

        return results.points

    def dieu_semantic_search(self, vector, van_ban_id: Optional[str] = None, chuong_id: Optional[str] = None, limit: int = 10):
        """Search for dieu documents similar to the vector"""
        
        # 1. Khởi tạo danh sách filters (Bỏ "payload." ở key)
        must_conditions = [
            models.FieldCondition(key="type", match=models.MatchValue(value="dieu"))
        ]
        
        # 2. Thêm điều kiện lọc theo van_ban_id nếu có
        if van_ban_id:
            must_conditions.append(
                models.FieldCondition(key="van_ban_id", match=models.MatchValue(value=van_ban_id))
            )
        
        # 3. Thêm điều kiện lọc theo chuong_id nếu có
        if chuong_id:
            must_conditions.append(
                models.FieldCondition(key="chuong_id", match=models.MatchValue(value=chuong_id))
            )
        
        results = self.client.query_points(
            collection_name=self.collection,
            query=vector,           # List các số thực (embedding)
            using="text_vector",    # 🔑 Phải khai báo tên vector nếu collection có nhiều loại vector
            query_filter=models.Filter(must=must_conditions),
            limit=limit,
            with_payload=True       # Trả về nội dung Điều luật
        )
        
        return results.points

