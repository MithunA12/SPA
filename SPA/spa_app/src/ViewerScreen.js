import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import PSGViewer from './PSGViewer';

// List of standardized channel names for mapping
const standardizedChannels = [
    "F3-M2", "F4-M1", "C3-M2", "C4-M1", "CZ-M1", "O1-M2", "O2-M1", "E1-M2",
    "E2-M1", "CHIN1-CHIN2", "LAT", "RAT", "SNORE", "PTAF", "AIRFLOW", "CHEST",
    "ABD", "IC", "EKG", "SaO2", "HR"
];

const ViewerScreen = ({ channelNames, filename, annotationFilename, setChannelMappings, setAnnotation }) => {
    const [channelMappings, setLocalChannelMappings] = useState({});
    const [annotation, setLocalAnnotation] = useState({ start: '', duration: '', description: '' });
    const [images, setImages] = useState([]);
    const [currentImageIndex, setCurrentImageIndex] = useState(0);
    const [isLoading, setIsLoading] = useState(false);
    const [spectrogram, setSpectrogram] = useState('');
    const [csvFile, setCsvFile] = useState('');
    const navigate = useNavigate();

    useEffect(() => {
        // Initialize channel mappings with default channel names
        const initialMappings = {};
        channelNames.forEach((channel, index) => {
            initialMappings[index] = channel;
        });
        setLocalChannelMappings(initialMappings);
    }, [channelNames]);

    useEffect(() => {
        fetchImages();  // Fetch images on component mount
    }, []);

    const fetchImages = () => {
        fetch('http://192.168.1.24:5000/images')
            .then(response => response.json())
            .then(data => {
                if (data.images) {
                    setImages(data.images);
                    if (data.spectrogram) {
                        setSpectrogram(data.spectrogram);
                    }
                    if (data.csv) {
                        setCsvFile(data.csv);
                    }
                } else {
                    console.error('Error fetching images:', data.error);
                }
            })
            .catch(error => {
                console.error('Error fetching images:', error);
            });
    };

    const generateImages = (index) => {
        setIsLoading(true);
        fetch('http://192.168.1.24:5000/generate_images', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            // Send necessary data to the backend
            body: JSON.stringify({ index, filename, channelMappings, annotation, annotationFilename })
        })
            .then(response => response.json())
            .then(data => {
                if (data.images) {
                    // Update images with new data from the backend
                    setImages(prevImages => [...prevImages, ...data.images]);
                    if (data.spectrogram) {
                        setSpectrogram(data.spectrogram);
                    }
                    if (data.csv) {
                        setCsvFile(data.csv);
                    }
                } else {
                    console.error('Error generating images:', data.error);
                }
                setIsLoading(false);
            })
            .catch(error => {
                console.error('Error generating images:', error);
                setIsLoading(false);
            });
    };

    const handleChannelChange = (index, value) => {
        // Update channel mapping when user selects a new channel
        setLocalChannelMappings(prevMappings => ({
            ...prevMappings,
            [index]: value
        }));
    };

    const handleAnnotationChange = (e) => {
        const { name, value } = e.target;
        // Update annotation state as user inputs data
        setLocalAnnotation(prevAnnotation => ({
            ...prevAnnotation,
            [name]: value
        }));
    };

    const handleSubmit = (event) => {
        event.preventDefault();
        // Update parent component's state and generate images
        setChannelMappings(channelMappings);
        setAnnotation(annotation);
        generateImages(Math.floor(images.length / 10) * 10);
    };

    const handleNextImage = () => {
        if (currentImageIndex < images.length - 2) {
            setCurrentImageIndex(currentImageIndex + 2);  // Move to the next set of images
        } else {
            generateImages(images.length / 2);  // Generate more images if at the end
        }
    };

    const handlePreviousImage = () => {
        if (currentImageIndex > 1) {
            setCurrentImageIndex(currentImageIndex - 2);  // Move to the previous set of images
        }
    };

    const downloadCSV = () => {
        // Creates a link to download the CSV file
        const link = document.createElement('a');
        link.href = `http://192.168.1.24:5000/download/${csvFile}`;
        link.download = csvFile;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    return (
        <div className="container">
            <h1 className="text-center mt-3">Sleep Stage PSG Viewer</h1>
            <button onClick={() => navigate('/')} className="btn btn-secondary mb-3">Back to Home</button>
            <div className="card mt-3">
                <div className="card-body">
                    <form onSubmit={handleSubmit}>
                        {/* Map over channel mappings to create select inputs */}
                        {Object.entries(channelMappings).map(([index, channel]) => (
                            <div key={index} className="mb-3 row">
                                <label className="form-label fw-bold col-3">{channelNames[index]}:</label>
                                <div className="col-8">
                                    <select
                                        value={channel}
                                        className="form-select"
                                        onChange={(e) => handleChannelChange(index, e.target.value)}
                                    >
                                        {standardizedChannels.map((ch, idx) => (
                                            <option key={idx} value={ch}>
                                                {ch}
                                            </option>
                                        ))}
                                    </select>
                                </div>
                            </div>
                        ))}
                        {/* Annotation inputs */}
                        <div className="mb-3 row">
                            <label className="form-label fw-bold col-3">Start Time:</label>
                            <div className="col-8">
                                <input
                                    type="number"
                                    name="start"
                                    className="form-control"
                                    value={annotation.start}
                                    onChange={handleAnnotationChange}
                                />
                            </div>
                        </div>
                        <div className="mb-3 row">
                            <label className="form-label fw-bold col-3">Duration:</label>
                            <div className="col-8">
                                <input
                                    type="number"
                                    name="duration"
                                    className="form-control"
                                    value={annotation.duration}
                                    onChange={handleAnnotationChange}
                                />
                            </div>
                        </div>
                        <div className="mb-3 row">
                            <label className="form-label fw-bold col-3">Description:</label>
                            <div className="col-8">
                                <input
                                    type="text"
                                    name="description"
                                    className="form-control"
                                    value={annotation.description}
                                    onChange={handleAnnotationChange}
                                />
                            </div>
                        </div>
                        <div className="text-center">
                            <button
                                type="submit"
                                className="btn btn-primary btn-md col-4"
                                style={{ marginRight: '8px' }}
                            >
                                Submit
                            </button>
                        </div>
                    </form>
                </div>
            </div>
            {/* Render the PSGViewer component */}
            <PSGViewer
                images={images}
                spectrogram={spectrogram}
                csvFile={csvFile}
                currentIndex={currentImageIndex}
                handlePreviousImage={handlePreviousImage}
                handleNextImage={handleNextImage}
                isLoading={isLoading}
                downloadCSV={downloadCSV}
            />
        </div>
    );
};

export default ViewerScreen;
