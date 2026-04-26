# Data Directory Structure

This directory contains the data resources required by the Engineering Drawing Intelligent Error Correction platform.

## Directory Structure

```
data/
├── drawings/              # User-uploaded drawing images (PNG/JPG/JPEG/GIF/BMP)
├── standard_drawings/     # Standard reference drawings for comparison
│   └── *.json            # Annotation files for each standard drawing
├── error_drawings/        # Error-annotated drawings for testing
├── error_labels/          # Error annotation text files
├── gb_standards/          # GB national standard PDF files
│   └── GBT 14665-2012.pdf
│   └── gbt14665_extracted.json
├── knowledge_base/        # Background knowledge JSON files (37 items)
│   └── knowledge_*.json
└── rl_experience/         # RL memory unit experience data (auto-generated)
```

## How to Add Your Own Resources

### 1. Standard Drawings

Place your standard engineering drawings in `data/standard_drawings/`:
- Supported formats: PNG, JPG, JPEG
- Recommended resolution: 1920x1080 or higher
- Optionally create a `.json` annotation file with the same name as the image

Example annotation file (`view.json`):
```json
{
  "type": "standard_drawing",
  "name": "view.png",
  "description": "Standard reducer assembly drawing"
}
```

### 2. GB Standards

Place GB standard PDF files in `data/gb_standards/`:
- The system will automatically extract text from PDF files
- Pre-extracted JSON files are also supported
- Example: `GBT 14665-2012.pdf` with corresponding `gbt14665_extracted.json`

### 3. Knowledge Base

Add knowledge items as JSON files in `data/knowledge_base/`:
- Each file should follow the `knowledge_N.json` naming convention
- Format:
```json
{
  "title": "Knowledge Item Title",
  "content": "Detailed content...",
  "source": "Source reference",
  "id": N
}
```

### 4. Error Drawings (for testing)

Place error-annotated drawings in `data/error_drawings/`:
- Use the `error_` prefix for filenames
- Create corresponding label files in `data/error_labels/`

## Important Notes

- DO NOT commit any personal or sensitive drawing files to public repositories
- The `rl_experience/` directory is auto-generated and should not be manually edited
- Image files in `drawings/` are for reference; actual uploads go to `uploads/`
