from preprocessing import get_sentence_embedding, transcribe_audio_file
from vdb import PineConeVdb
from config import get_config

config = get_config()
def run_text_model(text_api_key, audio_bytes, results, errors):
    try:
        text_engine = PineConeVdb(api_key=text_api_key)
        transcription = transcribe_audio_file(audio_bytes)
        embedding = get_sentence_embedding(transcription)
        text_result = text_engine.get_knn(
            k=1,
            vector=embedding.tolist(),
            namespace=config.VECTOR_DB_NAMESPACE
        )
        if not text_result.get("matches"):
            results['ayah'] = {
                "message": "Not found",
                "transcription": transcription,
            }
            return
        top_score = text_result["matches"][0].get("score", 0)
        if top_score < config.TEXT_MATCH_THRESHOLD:
            results['ayah'] = {
                "message": "Not found",
                "transcription": transcription,
            }
            results['sheikh'] = {"result": "No match found"}
            return
        filtered = [
            {
                "AyahNo": match["metadata"].get("AyahNo"),
                "SurahNo": match["metadata"].get("SurahNo")
            }
            for match in text_result["matches"]
            if match.get("metadata")
        ]
        if filtered:
            results['ayah'] = {
                "result": filtered[0],
                "transcription": transcription
            }
        else:
            results['ayah'] = {
                "message": "Not found",
                "transcription": transcription
            }
    except Exception as e:
        errors['ayah'] = str(e) 
