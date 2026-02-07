# Color Sampling Explorer

Interactive visualization demonstrating how different sampling methods affect outcomes when combining colored samples. A hands-on tool for teaching sampling theory, the Central Limit Theorem, and the impact of sample selection strategies.

## Live Demo

[Try the app](https://combining-colors.streamlit.app/)

## Features

**Interactive Controls:**
- Choose from 4 color palettes (RGB, CMY, RYB, Secondary)
- Adjust population composition (10-1000 samples per color)
- Select sample size percentage (10-100%)
- Compare 1-3 sampling methods simultaneously
- Control random seed for reproducibility

**Sampling Methods:**
- **Deterministic**: Takes first N% of each color (reproducible, potentially biased)
- **Random (no replacement)**: Each sample selected ≤1 time (classic random sampling)
- **Random (with replacement)**: Samples can be selected multiple times (bootstrap-style)

**Visualizations:**
- Row-wise comparison: sample selection and aggregated outcome side-by-side
- RGB values displayed directly on aggregate colors
- Opacity indicates selection frequency for replacement sampling
- Compressed layout prevents cropping and whitespace

**Built-in Documentation:**
- "How to Use" guide explaining all features
- "Key Statistical Insights" connecting to real applications
- Clear captions explaining what each visualization shows

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app"
5. Select your repo, branch, and set `app.py` as the main file
6. Click "Deploy"

## Educational Applications

This tool is designed for:
- Statistics courses teaching sampling methods
- Data science tutorials on reproducibility vs. randomness
- Demonstrating Central Limit Theorem visually
- Understanding bootstrap methods and resampling
- Exploring bias in deterministic selection strategies

## Key Insights

**Try this:**
1. Set unequal color counts (e.g., 500 first color, 300 second, 100 third)
2. Compare deterministic vs. random sampling at 50% sample size
3. Notice how deterministic always produces the same aggregate color
4. Change the random seed and watch random methods vary while deterministic stays constant
5. Enable only "Random (with replacement)" and watch darker dots appear where samples are selected multiple times
6. Switch between color palettes to see how different combinations mix (RGB vs CMY produce very different results!)

## Built With

- [Streamlit](https://streamlit.io/) - Web app framework
- [Plotly](https://plotly.com/python/) - Interactive visualizations
- [Pandas](https://pandas.pydata.org/) - Data manipulation
- [NumPy](https://numpy.org/) - Numerical computing

## Part of PixelProcess

This app is part of [PixelProcess](https://pixelprocess.org), a collection of interactive tools and resources for learning data science concepts.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Author

[Dexterous Data](https://dexterousdata.com) | [PixelProcess](https://pixelprocess.org)

## Contributing

Issues and pull requests welcome! This is designed as a teaching tool, so suggestions for improving clarity or educational value are especially appreciated.
