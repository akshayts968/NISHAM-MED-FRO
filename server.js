const express = require('express');
const mysql = require('mysql2');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
require('dotenv').config();
const { signup, login } = require('./authController');

// Auth Routes

const app = express();
app.use(cors());
app.use(express.json());

// --- Database Connection ---
const db = mysql.createPool({
    host: process.env.DB_HOST || 'localhost',
    user: process.env.DB_USER || 'root',
    password: process.env.DB_PASSWORD || '@Naina07122004',
    database: 'hospital_automanager'
});

// --- File Upload Configuration (Multer) ---
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, 'uploads/'); // Files will be saved in the /uploads folder
    },
    filename: (req, file, cb) => {
        cb(null, Date.now() + '-' + file.originalname);
    }
});
const upload = multer({ storage: storage });

app.post('/api/auth/signup', signup);
app.post('/api/auth/login', login);



// GET existing medical history to pre-fill the form
app.get('/api/medical-history/:userId', (req, res) => {
    const userId = req.params.userId;

    const sql = "SELECT * FROM medical_history WHERE user_id = ?";
    
    db.execute(sql, [userId], (err, results) => {
        if (err) {
            console.error("MySQL Error:", err.message);
            return res.status(500).json({ error: err.message });
        }
        
        // If the user has data, send the first row. If not, send null.
        if (results.length > 0) {
            res.json(results[0]);
        } else {
            res.json(null); 
        }
    });
});
app.post('/api/medical-history', (req, res) => {
    const { userId, age, sex, familyHistory, pastHeartProblem } = req.body;
    
    const sql = `INSERT INTO medical_history 
                 (user_id, age, sex, family_heart_history, past_heart_problem) 
                 VALUES (?, ?, ?, ?, ?)`;
    
    db.execute(sql, [userId, age, sex, familyHistory === 'yes' ? 1 : 0, pastHeartProblem], (err, result) => {
        if (err) return res.status(500).json({ error: err.message });
        res.json({ message: "Medical history saved successfully!" });
    });
});

app.post('/api/upload-report', upload.single('report'), (req, res) => {
    const { userId, reportType } = req.body;
    const filePath = req.file.path; // The location of the file on the server

    const sql = "INSERT INTO health_reports (user_id, report_type, file_path) VALUES (?, ?, ?)";
    
    db.execute(sql, [userId, reportType, filePath], (err, result) => {
        if (err) return res.status(500).json({ error: err.message });
        res.json({ message: "File uploaded and recorded in database!", filePath });
    });
});

// Start Server
const PORT = 5000;
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});