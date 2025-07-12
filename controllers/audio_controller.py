from audio_preprocessing import AudioPreprocessor
from vdb import AudioSimilaritySearch
from config import get_config

config = get_config()
def run_audio_model(audio_api_key, audio_bytes, results, errors):
    try:
        audio_preprocessor = AudioPreprocessor()
        audio_engine = AudioSimilaritySearch(api_key=audio_api_key)
        features = audio_preprocessor.preprocess_audio(audio_bytes)
        audio_result = audio_engine.search(features)
        if audio_result.get('matches'):
            best_match = audio_result['matches'][0]
            if best_match['score'] >= config.AUDIO_MATCH_THRESHOLD:
                results['sheikh'] = {
                    "match": best_match['metadata'].get(
                        'sheikh_name', 'Unknown'),
                    "confidence": float(best_match['score'])
                }
            else:
                results['sheikh'] = {"result": "No match found"}
        else:
            results['sheikh'] = {"result": "No match found"}
    except Exception as e:
        errors['sheikh'] = str(e) 
