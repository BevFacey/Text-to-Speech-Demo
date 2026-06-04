import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2';

let ttsPipeline = null;
let trainedVoiceProfile = null;

env.allowLocalModels = false;
env.backends.onnx.wasm.numThreads = navigator.hardwareConcurrency || 4;

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

function estimatePitchHz(pcm, sampleRate) {
    const minHz = 80;
    const maxHz = 350;
    const minLag = Math.floor(sampleRate / maxHz);
    const maxLag = Math.floor(sampleRate / minHz);
    let bestLag = minLag;
    let bestCorr = -Infinity;

    for (let lag = minLag; lag <= maxLag; lag++) {
        let corr = 0;
        const limit = pcm.length - lag;
        for (let i = 0; i < limit; i++) {
            corr += pcm[i] * pcm[i + lag];
        }
        if (corr > bestCorr) {
            bestCorr = corr;
            bestLag = lag;
        }
    }

    return sampleRate / bestLag;
}

function trainVoiceProfile(trainingInput, transcript = '') {
    const { pcm, sampleRate } = trainingInput;

    if (!pcm.length) {
        throw new Error('Reference audio is empty.');
    }

    let sumSq = 0;
    let peak = 0;
    for (let i = 0; i < pcm.length; i++) {
        const v = pcm[i];
        sumSq += v * v;
        peak = Math.max(peak, Math.abs(v));
    }

    const rms = Math.sqrt(sumSq / pcm.length);
    const durationSec = pcm.length / sampleRate;
    const words = transcript.trim() ? transcript.trim().split(/\s+/).length : 0;
    const speakingRateWps = words > 0 && durationSec > 0 ? words / durationSec : null;
    const pitchHz = estimatePitchHz(pcm, sampleRate);

    trainedVoiceProfile = {
        rms,
        peak,
        pitchHz,
        speakingRateWps,
    };

    return trainedVoiceProfile;
}

function resampleLinear(input, ratio) {
    if (!Number.isFinite(ratio) || ratio <= 0) return input;
    if (Math.abs(ratio - 1) < 0.01) return input;

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

function applyVoiceProfile(generatedAudio, generatedSampleRate) {
    if (!trainedVoiceProfile) return generatedAudio;

    const basePitch = 170;
    const baseRms = 0.12;

    const pitchRatio = Math.max(0.8, Math.min(1.25, trainedVoiceProfile.pitchHz / basePitch));
    const gain = Math.max(0.7, Math.min(1.5, trainedVoiceProfile.rms / baseRms));

    const pitched = resampleLinear(generatedAudio, pitchRatio);
    const styled = new Float32Array(pitched.length);

    for (let i = 0; i < pitched.length; i++) {
        styled[i] = Math.max(-1, Math.min(1, pitched[i] * gain));
    }

    return styled;
}

async function loadModels() {
    try {
        postMessage({ status: 'loading', message: 'Loading Transformers.js TTS model (this may take a moment)...' });
        ttsPipeline = await pipeline('text-to-speech', 'Xenova/mms-tts-eng', {
            quantized: true,
        });
        postMessage({ status: 'ready', message: 'System Ready. Please record reference voice.' });
    } catch (err) {
        postMessage({ status: 'error', message: 'Engine failure: ' + err.message });
    }
}

async function synthesize(text) {
    if (!ttsPipeline) return;
    if (!trainedVoiceProfile) {
        postMessage({ status: 'error', message: 'Record audio and train voice profile first.' });
        return;
    }

    postMessage({ status: 'processing', message: 'Generating speech with trained voice profile...' });

    try {
        const result = await ttsPipeline(text);
        const styledAudio = applyVoiceProfile(result.audio, result.sampling_rate);
        const wavData = toWavBuffer(styledAudio, result.sampling_rate);
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
            postMessage({ status: 'training', message: 'Training local voice profile from recording...' });

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

            const profile = trainVoiceProfile(trainingInput, e.data.transcript || '');
            postMessage({
                status: 'trained',
                message: `Voice profile trained (pitch ${Math.round(profile.pitchHz)} Hz). You can now generate speech.`
            });
        } catch (err) {
            postMessage({ status: 'error', message: 'Training error: ' + err.message });
        }
    } else if (e.data.action === 'synthesize') {
        await synthesize(e.data.text);
    }
};