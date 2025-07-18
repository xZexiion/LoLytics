import './App.css';

function App() {
    return (
        <div className="container">
            <div className="title">LoLytics</div>
            <div className="input-group">
                <input className="input-left" placeholder="Name" />
                <input className="input-right" placeholder="Tag" />
            </div>
            <button className="search-button"><span className='search-button-text'>Search</span></button>
        </div>
    );
}


export default App