def get_audio_bytes_from_request(req):
    if not req.files and not req.data:
        return None, {"error": "No audio data received"}, 400
    if 'audio' in req.files:
        audio_bytes = req.files['audio'].read()
    else:
        audio_bytes = req.get_data()
    if not audio_bytes:
        return None, {"error": "Empty audio data"}, 400
    return audio_bytes, None, None
