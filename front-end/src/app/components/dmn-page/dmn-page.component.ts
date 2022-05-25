import { Component, OnInit } from '@angular/core';
import { environment } from 'src/environments/environment';
import DmnJS from 'dmn-js';

@Component({
  selector: 'app-dmn-page',
  templateUrl: './dmn-page.component.html',
  styleUrls: ['./dmn-page.component.css']
})

export class DmnPageComponent implements OnInit {
  private startView = 0;
  private dmnViewer;
  constructor() {
  }

  // Parses one (1!) .csv's worth of sessionStorage, preparing it for POST to /rules
  processStorage() {
    const colNames = sessionStorage.getItem('words').split(',');
    // Continuous or categorical
    const inputTypes = sessionStorage.getItem('inputTypes').split(',');
    // input or output
    const colTypes = sessionStorage.getItem('col_types').split(',');
    const cleanedCols = [];
    let output = '';
    for (let i = 0; i < colNames.length; i++) {
      const colName = colNames[i];
      const colType = colTypes[i];
      const inputType = inputTypes[i];
      // If not input or output (AKA, exclude), do not use column
      if (!(['input', 'output'].includes(colType))) {
        continue;
      }
      // If output, put it aside
      if (colType === 'output') {
        output = colName;
        continue;
      }
      cleanedCols.push([colName, inputType]);
    }

    const cols = [];
    const model_id = sessionStorage.getItem('algorithms');
    const normalize_bool = sessionStorage.getItem('normalize_bool');
    const continuousCols = [];
    const fileString = sessionStorage.getItem('file');
    const fileName = sessionStorage.getItem('filename');
    const f = new File([fileString], fileName, { type: 'text/plain' });

    for (let i = 0; i < cleanedCols.length; i++) {
      const [colName, inputType] = cleanedCols[i];
      cols.push(colName);
      if (inputType === 'continue') {
        continuousCols.push(i);
      }

    }
    cols.push(output);
    return {
      json: {
        cols: [cols],
        output: [output],
        model_id: model_id,
        normalize_bool: normalize_bool,
        continuous_cols: [continuousCols]
      },
      files: [f]
    };
  }

// POST-request to /rules
// Returns a promise with json
// json contains a DMN-xml-string in "xml"
// and a list of accuracy scores in "accuracy"
  async requestDMN() {
    const parsedStorage = this.processStorage();
    const jsonMap = parsedStorage.json;
    const files = parsedStorage.files;
    const headers = { token: 'email@email.com' };

    const formData = new FormData();

    formData.append('json', JSON.stringify(jsonMap));
    formData.append('file', files[0]);

    const response = await fetch(environment.apiUri + 'rules', {
      method: 'POST',
      mode: 'cors',
      headers,
      body: formData
    });

    return response.json();
  }

// Function that starts after initialising the component.
// Makes a DMN-viewer, then uses response from /rules' xml output
// to show the DMN diagram
  ngOnInit() {
    // initialize DM Diagram Modeller.
    this.dmnViewer = new DmnJS({
      container: '#dmn-container' // This is the id of the html element we want to insert our modeller in.
    });
    // Fires when XML Import into DMN Modeller is complete.
    this.dmnViewer.on('import.done', ({ error, warnings }) => {
      console.log('Import Done event successfully emitted');
      if (!error) {
        // Draw the DMN model.
        // DMN Canvas will have multiple views (one view for each decision box/table)
        const view = this.dmnViewer.getViews()[this.startView];
        this.dmnViewer.open(view);
      } else {
        console.log(error);
      }
    });
    alert('The model is running, this could take a moment. Please have patience.');
    this.requestDMN().then(data => {
      this.dmnViewer.importXML(data.xml);
      console.log(data.accuracy);
    }).catch(_ => alert('500: Something went wrong at our api, please try again.'));
  }

  // Exports the DMN diagram's XML code
  // And downloads it to user
  exportDiagram() {
    this.dmnViewer.saveXML({ format: true }, (err, xml) => {
      if (err) {
        return console.error('Could not save DMN diagram', err);
      }
      const fileName = sessionStorage.getItem('name') + '.dmn';

      const f = new File([xml], fileName, { type: 'text/plain' });
      const url = window.URL.createObjectURL(f);

      const el = window.document.createElement('a');
      el.href = url;
      el.download = fileName;
      el.click();
    });
  }

}


