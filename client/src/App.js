import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import './App.css';
import logo from "./resources/logo.png";
import downloadimg from "./resources/download1.png";
import loader from "./resources/loader.gif";
import uploadimg from "./resources/upload1.png";
import { useDropzone } from 'react-dropzone';
import { BiChevronRight } from "react-icons/bi";
import { FaPlus, FaCheck } from 'react-icons/fa';

function App() {
  const [Name, setName] = useState();
  const [state, setState] = useState(0);
  const [url, setUrl] = useState();
  const [type,settype] = useState("YOLOv8n");
  const [typemenu,settypemenu] = useState(0);
  const fileInputRef = useRef();
  const canvasRef = useRef(null);
  const videoRef = useRef(null);
  const [isDraw, setIsDraw] = useState(0);
  const [aoipoly, setAoiPoly] = useState([]);
  const [npoints, setNpoints] = useState(0);
  const aoi = [];

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: "video/mp4, video/mov, video/avi",
    onDrop: (acceptedFiles) => {
      const files = acceptedFiles;
      setState(-1);
      let upvideo = new FormData();
      upvideo.append('file', files[0]);
      const filename = String(files[0].name);
      const fileUrl = URL.createObjectURL(files[0]); // Get the URL of the file
      setUrl(fileUrl);
      setName(filename);
      axios.post("http://localhost:5000/upload", upvideo).then(
        res => {
          setState(1);
        }
      );
    }
  });

  const Detectvideo = () => {
    if(aoipoly.length > 0){
    setState(-1);
    let detector = new FormData();
    detector.append('filename', Name);
    detector.append('aoi', aoipoly);
    axios.post("http://localhost:5000/convert", { filename: Name, model: type, aoi: aoipoly }).then(
      res => {
        console.log(res);
        setState(2);
      }
    );
  }
  };

  const modechange = () => {
    if (isDraw === 0) {
      setIsDraw(1);
    } else {
      setIsDraw(0);
      console.log(aoipoly);
    }
  };

  const Downloadfile = () => {
    setState(-1);
    axios({
      url: 'http://localhost:5000/downloadfile',
      method: 'GET',
      responseType: 'blob',
    }).then((response) => {
      const href = URL.createObjectURL(response.data);
      const link = document.createElement('a');
      link.href = href;
      link.setAttribute('download', 'video');
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(href);
      setState(2);
    });
  };

  useEffect(() => {
    const handleCanvasClick = (e) => {
      console.log(isDraw);
      if (isDraw === 0) {
        return;
      }

      const canvas = canvasRef.current;
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const offsetX = (e.clientX - rect.left) * scaleX;
      const offsetY = (e.clientY - rect.top) * scaleY;
      console.log("(" + offsetX + "," + offsetY + ")");
      aoi.push({ x: offsetX, y: offsetY });
      setAoiPoly(aoi);
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      ctx.strokeStyle = "red";
      ctx.fillStyle = "red";

      for (let i = 0; i < aoi.length; i++) {
        const point = aoi[i];
        ctx.beginPath();
        ctx.arc(point.x, point.y, 3, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();
      }
      setNpoints(aoi.length);
      if (aoi.length >= 3) {
        ctx.beginPath();
        ctx.moveTo(aoi[0].x, aoi[0].y);
        for (let i = 1; i < aoi.length; i++) {
          ctx.lineTo(aoi[i].x, aoi[i].y);
        }
        ctx.closePath();
        ctx.strokeStyle = "#1ac71a";
        ctx.stroke();
      }
    };

    console.log(canvasRef);
    const canvas = canvasRef.current;
    if (canvas) {
      canvas.addEventListener("click", handleCanvasClick);
    }

    return () => {
      if (canvas) {
        canvas.removeEventListener("click", handleCanvasClick);
      }
    };
  }, [isDraw]);

  return (
    <div className="App">
      <div className="headingdiv">
        <img src={logo} className="App-logo" alt="logo" />
        <h1 className="headingtxt">Vehicle Detection</h1>
      </div>
      {(state === -1) ? (
        <div>
          <img src={loader} alt="loader" className="Loader" />
          <h3 className="Loadingtxt">Loading</h3>
        </div>
      ) : (<div></div>)}
      {(state === 0) ? (
        <div>
          <div {...getRootProps()} className="downbox-cont">
            <input
              {...getInputProps()}
              type="file"
              accept="video/*"
              className="imginput"
              ref={fileInputRef}
            />
            <img src={uploadimg} className="downimg" alt="up" draggable="false" />
            <canvas className="canvasbroad" width={800} height={450} ref={canvasRef} hidden></canvas>
            <h1 className="droptxt">
              {isDragActive ? ("Drop the File Here") : ("Drop the File or Click Here to Upload")}
            </h1>
          </div>
        </div>
      ) : (
        <div></div>
      )}

      {(state === 1) ? (
        <div>
          <div className="hzbar">
            <div className="imgcont" style={{ position: "relative" }}>
              <div className="canvas-container">
                <video
                  muted
                  playsInline
                  controls
                  ref={videoRef}
                >
                  {url && <source src={url} type="video/mp4" />}
                </video>
                {isDraw ? (<canvas className="canvasbroad" width={800} height={450} ref={canvasRef}></canvas>) : (<canvas className="canvasbroad2" width={800} height={450} ref={canvasRef}></canvas>)}
              </div>
            </div>
            <div className="setbar">
              {typemenu?(<div className="popup-menu">
                <h5 onClick={() => { settype("YOLOv8n"); settypemenu(0); }}>YOLOv8n</h5>
                <h5 onClick={() => { settype("YOLOv8l"); settypemenu(0); }}>YOLOv8l</h5>
                <h5 onClick={() => { settype("Image Processing"); settypemenu(0); }}>Image Processing</h5>
                </div>):(<div></div>)}
              <h2 className="titleset">Settings</h2>
              <hr color="#eee"></hr>
              <div className="sectionp2">
                <div className="itemp2">
                  <h5 className="lstset1">Model</h5>
                  <h5 className="lstset2">{type}</h5>
                  <button
                    className={"addButton2"}
                    onClick={() => settypemenu(1)}
                  ><BiChevronRight size={16} className="lstset3" /></button>
                  
                </div>
                <div className="itemp2">
                  <h5 className="lstset1">Area of Interest</h5>
                  <h5 className="lstset22">{npoints} points</h5>
                  <button
                    className={"addButton"}
                    onClick={() => modechange()}
                  >
                    {isDraw ? <FaCheck /> : <FaPlus />}
                  </button>
                </div>
              </div>
              <div className="btnbox">
                <button type="button" className='detectbtn' onClick={Detectvideo}>Detect</button>
              </div>
            </div>
          </div>
        </div>
      ) : (<div></div>)}
      {(state === 2) ? (
        <div className="downbox-cont">
          <img src={downloadimg} className="downimg" alt="down" draggable="false" />
          <a className="downtxt" onClick={Downloadfile}>Click here to download output</a>
        </div>
      ) : (<div></div>)}
    </div>
  );
}

export default App;
