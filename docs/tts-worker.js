import { pipeline, env, AutoProcessor, AutoModel, Tensor } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2';

let ttsPipeline = null;
let speakerProcessor = null;
let speakerModel = null;
let trainedSpeakerEmbeddings = null;

env.allowLocalModels = false;
env.backends.onnx.wasm.numThreads = navigator.hardwareConcurrency || 4;

function toFloat32Audio(audioLike) {
    if (!audioLike) {
        throw new Error('Missing generated audio buffer.');
    }

    if (audioLike instanceof Float32Array) {
        return audioLike;
    }

    if (audioLike.data instanceof Float32Array) {
        return audioLike.data;
    }

    if (audioLike.data) {
        return new Float32Array(audioLike.data);
    }

    return new Float32Array(audioLike);
}

function sanitizeAndNormalizePcm(pcm) {
    if (!pcm.length) return pcm;

    const clean = new Float32Array(pcm.length);
    let peak = 0;
    for (let i = 0; i < pcm.length; i++) {
        const sample = Number.isFinite(pcm[i]) ? pcm[i] : 0;
        clean[i] = sample;
        const abs = Math.abs(sample);
        if (abs > peak) peak = abs;
    }

    // Protect against clipping and overly hot outputs.
    if (peak > 0.97) {
        const gain = 0.94 / peak;
        for (let i = 0; i < clean.length; i++) {
            clean[i] *= gain;
        }
    }

    // Short fades remove edge clicks at buffer boundaries.
    const fade = Math.min(320, Math.floor(clean.length / 10));
    for (let i = 0; i < fade; i++) {
        const g = i / fade;
        clean[i] *= g;
        clean[clean.length - 1 - i] *= g;
    }

    return clean;
}

// Encode Float32 PCM to a WAV ArrayBuffer so the main thread can play it directly.
function toWavBuffer(float32Pcm, sampleRate) {
    const channels = 1;
    const bitsPerSample = 16;
    const byteRate = sampleRate * channels * (bitsPerSample / 8);
    const blockAlign = channels * (bitsPerSample / 8);
    const dataSize = float32Pcm.length * 2;
    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);

    let offset = 0;
    const writeString = (str) => {
        for (let i = 0; i < str.length; i++) {
            view.setUint8(offset++, str.charCodeAt(i));
        }
    };

    writeString('RIFF');
    view.setUint32(offset, 36 + dataSize, true); offset += 4;
    writeString('WAVE');
    writeString('fmt ');
    view.setUint32(offset, 16, true); offset += 4;
    view.setUint16(offset, 1, true); offset += 2;
    view.setUint16(offset, channels, true); offset += 2;
    view.setUint32(offset, sampleRate, true); offset += 4;
    view.setUint32(offset, byteRate, true); offset += 4;
    view.setUint16(offset, blockAlign, true); offset += 2;
    view.setUint16(offset, bitsPerSample, true); offset += 2;
    writeString('data');
    view.setUint32(offset, dataSize, true); offset += 4;

    for (let i = 0; i < float32Pcm.length; i++) {
        const sample = Math.max(-1, Math.min(1, float32Pcm[i]));
        view.setInt16(offset, sample < 0 ? sample * 32768 : sample * 32767, true);
        offset += 2;
    }

    return buffer;
}

function fromWavBuffer(arrayBuffer) {
    const view = new DataView(arrayBuffer);

    if (view.getUint32(0, false) !== 0x52494646 || view.getUint32(8, false) !== 0x57415645) {
        throw new Error('Reference audio must be WAV format.');
    }

    let offset = 12;
    let sampleRate = 16000;
    let numChannels = 1;
    let bitsPerSample = 16;
    let pcmOffset = -1;
    let pcmSize = 0;

    while (offset + 8 <= view.byteLength) {
        const chunkId = view.getUint32(offset, false);
        const chunkSize = view.getUint32(offset + 4, true);
        offset += 8;

        if (chunkId === 0x666d7420) {
            numChannels = view.getUint16(offset + 2, true);
            sampleRate = view.getUint32(offset + 4, true);
            bitsPerSample = view.getUint16(offset + 14, true);
        } else if (chunkId === 0x64617461) {
            pcmOffset = offset;
            pcmSize = chunkSize;
            break;
        }

        offset += chunkSize + (chunkSize % 2);
    }

    if (pcmOffset < 0) {
        throw new Error('Invalid WAV data chunk.');
    }

    const bytesPerSample = bitsPerSample / 8;
    const totalSamples = Math.floor(pcmSize / bytesPerSample / numChannels);
    const mono = new Float32Array(totalSamples);

    let readOffset = pcmOffset;
    for (let i = 0; i < totalSamples; i++) {
        let sampleSum = 0;
        for (let ch = 0; ch < numChannels; ch++) {
            if (bitsPerSample === 16) {
                sampleSum += view.getInt16(readOffset, true) / 32768;
            } else if (bitsPerSample === 8) {
                sampleSum += (view.getUint8(readOffset) - 128) / 128;
            } else {
                throw new Error('Unsupported WAV bit depth: ' + bitsPerSample);
            }
            readOffset += bytesPerSample;
        }
        mono[i] = sampleSum / numChannels;
    }

    return { pcm: mono, sampleRate };
}

function resampleToRate(input, srcRate, dstRate) {
    if (srcRate === dstRate) return input;
    if (!input.length) return input;

    const ratio = srcRate / dstRate;
    const targetLength = Math.max(1, Math.floor(input.length / ratio));
    const output = new Float32Array(targetLength);

    for (let i = 0; i < targetLength; i++) {
        const srcIndex = i * ratio;
        const i0 = Math.floor(srcIndex);
        const i1 = Math.min(i0 + 1, input.length - 1);
        const frac = srcIndex - i0;
        output[i] = input[i0] * (1 - frac) + input[i1] * frac;
    }

    return output;
}

function l2Normalize(vector) {
    let sumSq = 0;
    for (let i = 0; i < vector.length; i++) {
        sumSq += vector[i] * vector[i];
    }

    const norm = Math.sqrt(sumSq);
    if (norm < 1e-8) return vector;

    const normalized = new Float32Array(vector.length);
    for (let i = 0; i < vector.length; i++) {
        normalized[i] = vector[i] / norm;
    }
    return normalized;
}

async function extractSpeakerEmbeddings(trainingInput) {
    if (!speakerProcessor || !speakerModel) {
        throw new Error('Speaker encoder is not ready yet.');
    }

    const { pcm, sampleRate } = trainingInput;
    if (!pcm.length) {
        throw new Error('Reference audio is empty.');
    }

    // Speaker encoder expects 16 kHz audio. We also cap long clips for speed.
    const pcm16k = resampleToRate(pcm, sampleRate, 16000);
    const maxSamples = 16000 * 12;
    const clipped = pcm16k.length > maxSamples ? pcm16k.slice(0, maxSamples) : pcm16k;

    const inputs = await speakerProcessor(clipped);
    const outputs = await speakerModel(inputs);

    if (!outputs.embeddings || !outputs.embeddings.data) {
        throw new Error('Speaker encoder returned no embeddings.');
    }

    const embeddingData = outputs.embeddings.data instanceof Float32Array
        ? outputs.embeddings.data
        : new Float32Array(outputs.embeddings.data);

    const normalized = l2Normalize(embeddingData);
    return new Tensor('float32', normalized, [1, normalized.length]);
}

async function loadModels() {
    try {
        postMessage({ status: 'loading', message: 'Loading high-quality SpeechT5 voice-cloning model...' });
        ttsPipeline = await pipeline('text-to-speech', 'Xenova/speecht5_tts', {
            quantized: false,
        });

        postMessage({ status: 'loading', message: 'Loading speaker embedding encoder...' });
        speakerProcessor = await AutoProcessor.from_pretrained('Xenova/wavlm-base-plus-sv');
        speakerModel = await AutoModel.from_pretrained('Xenova/wavlm-base-plus-sv', {
            quantized: true,
        });

        postMessage({ status: 'ready', message: 'System ready. Record reference voice to train speaker identity.' });
    } catch (err) {
        postMessage({ status: 'error', message: 'Engine failure: ' + err.message });
    }
}

async function synthesize(text) {
    if (!ttsPipeline) return;
    if (!trainedSpeakerEmbeddings) {
        postMessage({ status: 'error', message: 'Record and train reference voice first.' });
        return;
    }

    postMessage({ status: 'processing', message: 'Generating speech with speaker-conditioned synthesis...' });

    try {
        const result = await ttsPipeline(text, {
            speaker_embeddings: trainedSpeakerEmbeddings,
        });

        if (!result || !result.audio || !result.sampling_rate) {
            throw new Error('Model returned invalid audio output.');
        }

        const rawAudio = toFloat32Audio(result.audio);
        const cleanedAudio = sanitizeAndNormalizePcm(rawAudio);
        const wavData = toWavBuffer(cleanedAudio, result.sampling_rate);
        postMessage({ status: 'complete', audioData: wavData }, [wavData]);
    } catch (err) {
        postMessage({ status: 'error', message: 'Inference error: ' + err.message });
    }
}

self.onmessage = async (e) => {
    if (e.data.action === 'init') {
        await loadModels();
    } else if (e.data.action === 'train') {
        try {
            postMessage({ status: 'training', message: 'Extracting speaker embeddings from reference audio...' });

            let trainingInput;
            if (e.data.refPcmData && e.data.refSampleRate) {
                trainingInput = {
                    pcm: new Float32Array(e.data.refPcmData),
                    sampleRate: e.data.refSampleRate,
                };
            } else if (e.data.refAudioData) {
                trainingInput = fromWavBuffer(e.data.refAudioData);
            } else {
                throw new Error('Missing training audio data.');
            }

            trainedSpeakerEmbeddings = await extractSpeakerEmbeddings(trainingInput);
            postMessage({
                status: 'trained',
                message: 'Speaker embedding trained. Generate speech to hear cloned voice identity.'
            });
        } catch (err) {
            postMessage({ status: 'error', message: 'Training error: ' + err.message });
        }
    } else if (e.data.action === 'synthesize') {
        await synthesize(e.data.text);
    }
};