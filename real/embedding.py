from FlagEmbedding import FlagModel

class BGEEmbedding:
    def __init__(self, 
                 model_name="BAAI/bge-m3", 
                 device="cpu", 
                 fp16=False):
        """
        Khởi tạo model BGE-M3 để embedding.
        """
        self.model = FlagModel(
            model_name_or_path=model_name,
            device=device,
            fp16=fp16
        )

    def embed(self, texts):
        """
        Tạo embedding cho văn bản.
        
        texts: str hoặc list[str]
        Trả về: list vector (list[float]) nếu input là list
                 vector (list[float]) nếu input là str
        """
        # Nếu chỉ 1 câu, gói vào list
        single_input = False
        if isinstance(texts, str):
            texts = [texts]
            single_input = True

        # Encode
        vectors = self.model.encode(texts)

        if single_input:
            return vectors[0]
        return vectors
