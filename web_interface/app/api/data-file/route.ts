import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

// Define the data directory relative to the project root
// For Next.js, the process.cwd() is the root of the project where the package.json is located
// So we need to set the data path properly
const DATA_DIR = path.join(process.cwd(), '../data');

// Log the path for debugging
console.log('DATA_DIR path:', DATA_DIR);

/**
 * GET handler for data file retrieval
 * 
 * This endpoint serves the content of data files as plain text for viewing in the browser.
 * It expects a 'path' query parameter with the relative path of the file to retrieve.
 */
export async function GET(request: NextRequest) {
  try {
    // Get the requested file path from the query parameters
    const searchParams = request.nextUrl.searchParams;
    const filePath = searchParams.get('path');

    console.log('Requested file path:', filePath);
    
    // Validate the file path
    if (!filePath) {
      return NextResponse.json(
        { error: 'File path is required' },
        { status: 400 }
      );
    }

    // Ensure the path does not contain any path traversal attempts
    const normalizedPath = path.normalize(filePath).replace(/^(\.\.(\/|\\|$))+/, '');
    
    // Try both the data directory path and a direct path from the root
    let fullPath = path.join(DATA_DIR, normalizedPath);
    console.log('Attempting to read from:', fullPath);
    
    if (!fs.existsSync(fullPath)) {
      // Try alternate path directly from project root
      fullPath = path.join(process.cwd(), normalizedPath);
      console.log('Alternate path attempt:', fullPath);
      
      if (!fs.existsSync(fullPath)) {
        return NextResponse.json(
          { error: 'File not found', requested: normalizedPath, paths_tried: [path.join(DATA_DIR, normalizedPath), fullPath] },
          { status: 404 }
        );
      }
    }

    // Read the file content
    const fileContent = fs.readFileSync(fullPath, 'utf8');

    // Determine the content type based on file extension
    const ext = path.extname(fullPath).toLowerCase();
    let contentType = 'text/plain';
    
    switch (ext) {
      case '.json':
        contentType = 'application/json';
        break;
      case '.csv':
        contentType = 'text/csv';
        break;
      case '.py':
        contentType = 'text/x-python';
        break;
    }

    // Return the file content with the appropriate content type
    return new NextResponse(fileContent, {
      status: 200,
      headers: {
        'Content-Type': contentType,
        'Content-Disposition': `inline; filename="${path.basename(fullPath)}"`,
      },
    });
  } catch (error) {
    console.error('Error serving data file:', error);
    return NextResponse.json(
      { error: 'Internal server error', details: String(error) },
      { status: 500 }
    );
  }
} 