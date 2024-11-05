const express = require('express');
const multer = require('multer');
const pdfParse = require('pdf-parse');
const axios = require('axios');
const fs = require('fs');
const path = require('path');

const app = express();
const port = 3000;

// Middleware to serve static files (HTML)
app.use(express.static(path.join(__dirname, 'public')));

// Configure multer for file uploads
const upload = multer({ dest: 'uploads/' });

// Endpoint to handle PDF upload
app.post('/upload', upload.single('pdfFile'), async (req, res) => {
    try {
        const pdfPath = req.file.path;

        // Extract text from PDF
        const dataBuffer = fs.readFileSync(pdfPath);
        const pdfData = await pdfParse(dataBuffer);

        // Extracted text from the PDF
        const extractedText = pdfData.text;

        // Call the Python API to generate scenes (Qwen 2.5 Instruct)
        const response = await axios.post('http://localhost:8000/generate_script', {
            text: extractedText
        });

        const generatedScenes = response.data;

        // For each scene, generate an image, video, and audio
        const images = [];
        const videos = [];
        const audios = [];
        for (let scene of generatedScenes.scenes) {
            // Generate Image
            const imageResponse = await axios.post('http://localhost:8000/generate_image', {
                prompt: scene.image_prompt
            });
            const imagePath = imageResponse.data.image_path;
            images.push(imagePath);

            // Generate Video using the image and video prompt
            const videoResponse = await axios.post('http://localhost:8000/generate_video', {
                image_path: imagePath,
                video_prompt: scene.video_prompt
            });
            videos.push(videoResponse.data.video_path);

            // Generate Audio using the scene script
            const audioResponse = await axios.post('http://localhost:8000/generate_audio', {
                script: scene.script
            });
            audios.push(audioResponse.data.audio_path);
        }

        // Call the RIFE API to enhance video transitions
        const enhancementResponse = await axios.post('http://localhost:8000/enhance_video', {
            video_paths: videos
        });

        const enhancedVideoPaths = enhancementResponse.data.enhanced_video_path;

        // Call the FFmpeg API to merge audio with video
        const mergeResponse = await axios.post('http://localhost:8000/merge_audio_video', {
            video_paths: enhancedVideoPaths,
            audio_paths: audios
        });

        const mergedVideos = mergeResponse.data.merged_videos;

        // Send the final merged videos back to the client
        res.json({ merged_videos: mergedVideos });

    } catch (error) {
        console.error('Error processing the PDF:', error);
        res.status(500).send('Error processing the PDF');
    }
});

// Endpoint to handle product image upload for captioning
app.post('/upload_image', upload.single('imageFile'), async (req, res) => {
    try {
        const imagePath = req.file.path;

        // Read the image file
        const image = fs.readFileSync(imagePath);

        // Send image to Python API (Florence 2) for captioning
        const response = await axios.post('http://localhost:8000/generate_caption', image, {
            headers: {
                'Content-Type': 'multipart/form-data'
            }
        });

        const caption = response.data.caption;

        // Send the generated caption back to the client
        res.json({ caption: caption });

    } catch (error) {
        console.error('Error generating image caption:', error);
        res.status(500).send('Error generating image caption');
    }
});

app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});
