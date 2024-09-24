import React, { useState } from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Home from './Home';
import ViewerScreen from './ViewerScreen';
import './App.css';

function App() {
  const [channelNames, setChannelNames] = useState([]);
  const [filename, setFilename] = useState('');
  const [annotationFilename, setAnnotationFilename] = useState('');
  const [channelMappings, setChannelMappings] = useState({});
  const [annotation, setAnnotation] = useState({ start: '', duration: '', description: '' });

  return (
    <Router>
      <div className="container">
        <Routes>
          <Route path="/" element={<Home setChannelNames={setChannelNames} setFilename={setFilename} setAnnotationFilename={setAnnotationFilename} />} />
          <Route path="/viewer" element={<ViewerScreen channelNames={channelNames} filename={filename} annotationFilename={annotationFilename} setChannelMappings={setChannelMappings} setAnnotation={setAnnotation} />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
