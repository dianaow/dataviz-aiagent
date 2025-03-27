# Advanced Materials Data Analyzer

An AI-driven tool for analyzing, visualizing, and interpreting materials metrology data, with a focus on battery testing.

## Features

- **Automated Data Classification**: Uses AI to identify the type of metrology data (battery cycling, XRD, Raman spectroscopy, etc.)
- **Specialized Visualizations**: Generates context-specific plots based on the data type
- **Advanced Analysis**: Performs peak fitting, capacity retention calculations, and other analytical procedures
- **Intelligent Insights**: Provides interpretations and comparisons with reference data

## Supported Data Types

- Battery Cycling Data
- X-ray Diffraction (XRD)
- Raman Spectroscopy
- Electrochemical Impedance Spectroscopy (EIS)
- Thermal Analysis
- Surface Area Analysis
- Generic Data (fallback)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/advanced-materials-data-analyzer.git
   cd advanced-materials-data-analyzer
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up your environment variables:
   - Copy `.env.example` to `.env`
   - Add your OpenAI API key to `.env` for LLM-based data classification

## Usage

1. Start the application:
   ```
   python app.py
   ```

2. Open your web browser and navigate to `http://localhost:8050`

3. Upload your data file (CSV, Excel, or text format)

4. Optionally provide a hint about the data type

5. Click "Process Data" to analyze

## Data Flow

1. **Data Ingestion**: Upload a CSV, Excel, or text file
2. **Data Classification**: AI identifies the type of metrology data
3. **Data Processing**: The system performs type-specific data cleaning and preparation
4. **Visualization**: Interactive plots are generated based on the data type
5. **Analysis**: The system performs peak fitting, calculates key metrics, and provides insights

## Examples

### Battery Cycling Data
- Displays capacity retention, voltage hysteresis, differential capacity analysis
- Calculates Coulombic efficiency and capacity fade rates

### X-ray Diffraction
- Displays diffraction patterns, performs peak fitting
- Estimates crystallite sizes based on peak widths

### Raman Spectroscopy
- Visualizes spectral data, identifies characteristic peaks
- For carbon materials, calculates ID/IG ratios and identifies D and G bands

## Troubleshooting

- **Classification Errors**: If the system incorrectly identifies your data type, you can manually select the correct type.
- **Visualization Issues**: Try different plot types from the dropdown menu for alternative visualizations.
- **Missing OpenAI API Key**: The system will fall back to rule-based classification if the OpenAI API key is not provided.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 