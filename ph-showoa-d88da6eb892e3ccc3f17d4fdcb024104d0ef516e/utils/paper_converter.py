import json
from pathlib import Path

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat

# 1. Import AcceleratorOptions to limit threads
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions

# 2. Import the lightweight PyPdfium2 backend
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

def convert_pdf_to_json(pdf_path: str, output_json_path: str):
    pipeline_options = PdfPipelineOptions()
    
    # Keep memory optimizations
    pipeline_options.generate_page_images = False       
    pipeline_options.generate_picture_images = True    
    pipeline_options.do_ocr = True                      
    pipeline_options.images_scale = 1.0
    # NEW: Constrain threads. 
    # By default, Docling tries to process as many pages simultaneously as you have CPU cores.
    # Forcing it to 1 or 2 threads drastically reduces RAM/VRAM spikes.
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=2, 
        device="auto"
    )

    # NEW: Inject the PyPdfiumDocumentBackend
    # This bypasses the heavy C++ parser that is throwing the std::bad_alloc errors.
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=PyPdfiumDocumentBackend
            )
        }
    )

    print(f"Processing: {pdf_path}...")

    # Run conversion
    result = converter.convert(pdf_path)
    
    # Export to dictionary
    doc_dict = result.document.export_to_dict()

    # Save to JSON
    output_path = Path(output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(doc_dict, f, indent=2, ensure_ascii=False)

    print(f"Successfully saved JSON to: {output_path}")

if __name__ == "__main__":
    convert_pdf_to_json("journal.pone.0343262.pdf", "paper.json")