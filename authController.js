const mysql = require('mysql2');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
require('dotenv').config();

const db = mysql.createPool({
    host: 'localhost',
    user: 'root',
    password: '@Naina07122004',
    database: 'hospital_automanager'
});

const signup = async (req, res) => {
    const { firstName, lastName, email, mobile, password } = req.body;

    try {
        // Hash the password before saving
        const salt = await bcrypt.genSalt(10);
        const hashedPassword = await bcrypt.hash(password, salt);

        const sql = "INSERT INTO users (first_name, last_name, email, mobile, password) VALUES (?, ?, ?, ?, ?)";
        db.execute(sql, [firstName, lastName, email, mobile, hashedPassword], (err, result) => {
            if (err) {
                if (err.code === 'ER_DUP_ENTRY') return res.status(400).json({ error: "Email already exists" });
                return res.status(500).json({ error: err.message });
            }
            res.status(201).json({ message: "User registered successfully!" });
        });
    } catch (err) {
        res.status(500).json({ error: "Server error" });
    }
};

// --- 2. LOGIN API ---
const login = async (req, res) => {
    const { email, password } = req.body;

    const sql = "SELECT * FROM users WHERE email = ?";
    db.execute(sql, [email], async (err, results) => {
        if (err) return res.status(500).json({ error: err.message });
        if (results.length === 0) return res.status(400).json({ error: "Invalid Credentials" });

        const user = results[0];

        // Compare entered password with hashed password in DB
        const isMatch = await bcrypt.compare(password, user.password);
        if (!isMatch) return res.status(400).json({ error: "Invalid Credentials" });

        // Create JWT Token
        const token = jwt.sign(
            { id: user.id, email: user.email },
            process.env.JWT_SECRET || 'your_super_secret_key',
            { expiresIn: '1h' }
        );

        res.json({
            token,
            user: { id: user.id, firstName: user.first_name, lastName: user.last_name }
        });
    });
};

module.exports = { signup, login };