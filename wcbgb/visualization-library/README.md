# Sensemaker Visualization Components

> **Note:** This package is **archival, unmaintained, and read-only**, preserved as part of the **What Could BG Be?** case study (see [../README.md](../README.md)).

A collection of reusable visualization components for displaying Sensemaker data, built with D3.js and Web Components.

### Installation
```bash
npm install
```

### Development
To run the Storybook development environment:
```bash
npm run storybook
```
This will start the Storybook server at `http://localhost:6006`. You can view and interact with all components in isolation.

> **Note:** The Storybook stories use sample data from the `stories/data` directory. To view all stories correctly, ensure you have both a `comments.json` and a `summary.json` file present in `stories/data/`.

### Building the Package
To build the package:
```bash
npm run build
```
The compiled files will be output to the `dist/` directory.

### Building Storybook Docs Locally
To build a local static Storybook site:
```bash
npm run build-storybook
```
The static site will be output to the `storybook-static/` directory.

## Data Source and License

The data used in this demo was gathered using the [Polis software](https://compdemocracy.org/Polis/) and is sub-licensed under CC BY 4.0 with Attribution to The Computational Democracy Project. The data and more information about how the data was collected can be found at the following link:

[https://github.com/compdemocracy/openData/tree/master/american-assembly.bowling-green](https://github.com/compdemocracy/openData/tree/master/american-assembly.bowling-green)
