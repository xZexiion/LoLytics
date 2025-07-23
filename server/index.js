import '../data/game.js';
import express from 'express';

const app = express();
const PORT = 3000;

app.get('/match_history/', (req, res) => {
    const { name, tag } = req.query;
    res.send(`Fetching match history for ${name}#${tag}`)
});

app.listen(PORT, () => {
    console.log(`Server is running at http://localhost:${PORT}`);
});
