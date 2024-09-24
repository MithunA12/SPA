import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

const Home = ({ setChannelNames, setFilename, setAnnotationFilename }) => {

    const [file, setFile] = useState(null);
    const [annotationFile, setAnnotationFile] = useState(null);
    const navigate = useNavigate();

    // Handle change event for the main file input
    const handleFileChange = (event) => {
        const selectedFile = event.target.files[0];
        if (selectedFile) {
            setFile(selectedFile);
        }
    };

    // Handle change event for the annotation file input
    const handleAnnotationFileChange = (event) => {
        const selectedFile = event.target.files[0];
        if (selectedFile) {
            setAnnotationFile(selectedFile);
        }
    };

    // Handle form submission to upload files
    const handleSubmit = (event) => {
        event.preventDefault();

        if (!file) {
            alert('Please select a file to upload.');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);
        if (annotationFile) {
            formData.append('annotation_file', annotationFile);
        }

        // POST request to upload files
        fetch('http://192.168.1.24:5000/upload', {
            method: 'POST',
            body: formData
        })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    console.error('Error uploading file:', data.error);
                } else {
                    // Updates parent state with response data and navigate to viewer
                    setChannelNames(data.channelNames);
                    setFilename(data.filename);
                    setAnnotationFilename(data.annotationFilename || '');
                    navigate('/viewer');
                }
            })
            .catch(error => {
                console.error('Error uploading file:', error);
            });
    };

    return (
        <div className="container">
            <h1 className="text-center mt-3">Upload PSG Files</h1>
            <div className="card mt-3">
                <div className="card-body">
                    <form onSubmit={handleSubmit}>
                        <div className="mb-3">
                            <label className="form-label fw-bold">Upload EDF File:</label>
                            <input type="file" className="form-control" onChange={handleFileChange} />
                        </div>
                        <div className="mb-3">
                            <label className="form-label fw-bold">Upload Annotations File (Optional):</label>
                            <input type="file" className="form-control" onChange={handleAnnotationFileChange} />
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
        </div>
    );
};

export default Home;
