import React from 'react';

//PSGViewer component that displays PSG, sleep stages, and spectrogram, with navigation controls and CSV download
function PSGViewer({ images, spectrogram, csvFile, currentIndex, handlePreviousImage, handleNextImage, isLoading, downloadCSV }) {

    //Function to determine the type of image {raw PSG data (naming I used for plots created from edf), sleep stage signal, or spectrogram} based on the filename
    const getImageType = (filename) => {
        if (!filename) return 'Unknown';
        if (filename.includes('raw_image')) {
            return 'Raw PSG Data';
        } else if (filename.includes('sleep_stage_image')) {
            return 'Sleep Stage Signal';
        } else if (filename.includes('spectrogram')) {
            return 'Spectrogram';
        } else {
            return 'Unknown'; // defaults to 'Unknown' if none of the above match
        }
    };

    //Gets the corresponding raw and sleep stage images for the current index
    const getCorrespondingImages = (currentIndex) => {
        const rawImage = images.find(img => img.includes(`raw_image_${Math.floor(currentIndex / 2)}`)); // finds raw image for the current index
        const sleepStageImage = images.find(img => img.includes(`sleep_stage_image_${Math.floor(currentIndex / 2)}`)); // does the same for sleep stage image
        return { rawImage, sleepStageImage };
    };

    const { rawImage, sleepStageImage } = getCorrespondingImages(currentIndex);

    return (
        <div className="text-center mt-3">
            {/* Displays the sleep stage image if available already*/}
            {sleepStageImage && (
                <div>
                    <img
                        src={`http://192.168.1.24:5000/images/${sleepStageImage}`}
                        alt={`Sleep Stage Image ${currentIndex + 1}`}
                        style={{ width: '30%', height: '30%' }}
                    />
                </div>
            )}

            {/* checks if there are images available */}
            {images.length > 0 && (
                <div>
                    <div style={{ display: 'flex', alignItems: 'center' }}>
                        {/* displays the raw PSG image if available */}
                        {rawImage && (
                            <div>
                                <img
                                    src={`http://192.168.1.24:5000/images/${rawImage}`}
                                    alt={`PSG Image ${currentIndex + 1}`}
                                    style={{ width: '80%', height: '80%' }}
                                />
                            </div>
                        )}
                    </div>

                    {/* Navigation buttons for Previous and Next images */}
                    <div className="mt-3">
                        <button onClick={handlePreviousImage} disabled={currentIndex === 0} className="btn btn-secondary me-2">
                            Previous Image
                        </button>
                        <button onClick={handleNextImage} disabled={isLoading} className="btn btn-primary">
                            {isLoading ? "Loading..." : "Next Image"}
                        </button>
                    </div>

                    {/* Display the spectrogram if available */}
                    {spectrogram && (
                        <div className="mt-3">
                            <p>Spectrogram</p>
                            <img
                                src={`http://192.168.1.24:5000/images/${spectrogram}`}
                                alt="Spectrogram"
                                style={{ width: '25%', height: '25%' }}
                            />
                        </div>
                    )}

                    {/* Download button for the CSV file */}
                    {csvFile && (
                        <div className="mt-3">
                            <button onClick={downloadCSV} className="btn btn-primary">Download CSV</button>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}

export default PSGViewer;
