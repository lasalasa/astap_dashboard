# ✈️ ASTAPD - Aviation Safety Trends Analysis and Predictive Dashboard

**ASTAPD** is an interactive **Dash-based dashboard** designed to analyze **aviation safety trends** and **predict human factors** contributing to incidents. It integrates **machine learning models** and **historical data** from ASRS and NTSB, visualizing trends for improved aviation safety insights.

---

## 📌 Features
- **📊 Data Visualization**: Graphs and charts for historical aviation safety trends
- **🤖 Machine Learning Predictions**: Predictive insights using LSTMs and human factors modeling
- **🛠 Modular Dash App**: Scalable and maintainable architecture
- **🌐 Responsive UI**: Powered by Bootstrap for seamless navigation
- **🔍 Multi-Page Support**: Separate pages for analysis, predictions, and reports
- **🚀 Ready for Deployment**: Supports Gunicorn, Docker, and cloud hosting

---

## 🔧 Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/astapd-dashboard.git
cd astapd-dashboard
```

### 2️⃣ Install Dependencies
Use `pip` to install required Python packages.
```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing, manually install dependencies:
```bash
pip install dash dash-bootstrap-components flask gunicorn pandas numpy plotly
```

---

## ▶️ Running the ASTAPD Dashboard

### 🛠 Development Mode
To start the dashboard in development mode:
```bash
python app.py
```
Access the app at: **[http://127.0.0.1:8050](http://127.0.0.1:8050/)**

---

### 🚀 Production Deployment
Coming soon, stay tuned.
<!-- #### **Option 1: Gunicorn (Recommended)**
For a production-ready setup:
```bash
gunicorn server:server
```

#### **Option 2: Docker**
To run the app inside a Docker container:

1. **Build the image**
   ```bash
   docker build -t astapd .
   ```
2. **Run the container**
   ```bash
   docker run -p 8050:8050 astapd
   ```

Access the app at: **[http://localhost:8050](http://localhost:8050/)** -->

---

## 📁 Project Structure
```
astapd/
│-- app.py               # Main Dash app
│-- server.py            # Flask-Gunicorn server
│-- components/
│   │-- navbar.py        # Navigation bar
│   │-- layout.py        # Main layout
│   │-- graphs.py        # Graph generation logic
│   │-- ml_predictions.py # ML model integration
│-- assets/              # Static files (CSS, JS)
│-- pages/               # Multi-page support
│   │-- trends.py        # Historical trends analysis
│   │-- predictions.py   # Predictive modeling results
│   │-- reports.py       # Custom reports and exports
│-- data/                # Processed and raw aviation data
│-- models/              # Machine learning models
│-- requirements.txt     # Python dependencies
│-- Dockerfile           # Docker support
│-- README.md            # Project documentation
```

---

## 🛠 Customization
- Modify `components/layout.py` to update the dashboard’s layout
- Add new visualizations in `components/graphs.py`
- Integrate custom ML models in `components/ml_predictions.py`
- Store and preprocess aviation safety datasets in the `data/` folder

---

## 📊 Data Sources
ASTAPD utilizes aviation safety data from:
- **ASRS (Aviation Safety Reporting System)**
- **NTSB (National Transportation Safety Board)**
- **FAA (Federal Aviation Administration)**
- **Custom LSTM Models for Predicting Human Factors**

---

## ⚡ Future Enhancements
- 🔹 **Interactive Filters** for dynamic trend exploration
- 🔹 **Real-time Data Feeds** integration with aviation databases
- 🔹 **User Authentication** for personalized dashboards
- 🔹 **Downloadable Reports** in PDF/CSV formats

---

## 📖 References
- Dash DataTables: https://dash.plotly.com/datatable
- Aviation Safety Network: https://flightsafety.org/toolkits-resources/aviation-safety-network/
- IMDB Dashboard Example: https://github.com/Mahmoud2227/IMDB-Dashboard/blob/master/app.py
- Improving Dash UI: https://towardsdatascience.com/3-easy-ways-to-make-your-dash-application-look-better-3e4cfefaf772

---

## 📄 License
This project is licensed under the MIT License.

---

### 🎯 Author
**Lasantha Lakmal**  
[GitHub](https://github.com/lasalasa) 

<!-- | [Website](https://yourwebsite.com) -->

---
<!--
🚀 **Elevating Aviation Safety with Data and AI!** 🚀
```

---

### **How to Use This?**
1. Replace `yourusername` with your actual **GitHub username**.
2. Update `[Website](https://yourwebsite.com)` if you have a personal or company website.
3. Expand the **Future Enhancements** section based on new features you plan to add. -->

This `README.md` provides **clear documentation, easy installation steps, and a structured overview** of ASTAP Dshboard. Let me know if you want any refinements! 🚀 ✈️
