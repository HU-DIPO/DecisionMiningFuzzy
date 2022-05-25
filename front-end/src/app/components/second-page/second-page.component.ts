import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-second-page',
  templateUrl: './second-page.component.html',
  styleUrls: ['./second-page.component.css']
})
export class SecondPageComponent implements OnInit {

  constructor() {
  }

  // Build an element for radio buttons.
  // Used to build a form-box
  buildRadioButton(value: string, clssPrefix: string, checked: boolean, idx: number): Element {
    const rbs = document.createElement('input');
    rbs.name = 'results';
    rbs.type = 'radio';
    rbs.value = value;
    rbs.id = clssPrefix + idx.toString();
    rbs.className = clssPrefix + 'button';
    if (checked) { rbs.setAttribute('checked', 'checked'); }

    return rbs;
  }
// Build the "form-box-first" element
// Contains three radio buttons for selecting "input", "output", "exclude" per column
// Also with labels
  buildFirstBox(idx: number): Element {
    const firstBox = document.createElement('form');
    firstBox.className = 'form-box-first';
    // Build input button
    firstBox.appendChild(this.buildRadioButton('input', 'input', true, idx));
    const labelInput = document.createElement('label');
    labelInput.setAttribute('for', 'input' + idx.toString());
    labelInput.innerHTML = 'Input';
    firstBox.appendChild(labelInput);

    // Build output
    firstBox.appendChild(this.buildRadioButton('output', 'output', false, idx));
    const labelOutput = document.createElement('label');
    labelOutput.setAttribute('for', 'output' + idx.toString());
    labelOutput.innerHTML = 'Output';
    firstBox.appendChild(labelOutput);

    // Build exclude button
    firstBox.appendChild(this.buildRadioButton('exclude', 'exclude', false, idx));
    const labelExclude = document.createElement('label');
    labelExclude.setAttribute('for', 'exclude' + idx.toString());
    labelExclude.innerHTML = 'Exclude';
    firstBox.appendChild(labelExclude);
    firstBox.append(document.createElement('br'));


    return firstBox;
  }
// Build the "form-box-second" element
// Contains two radio buttons for selecting "continue" or "categoriaal" per column
// Also with labels
  buildSecondBox(idx: number): Element {
    const secondBox = document.createElement('form');
    secondBox.className = 'form-box-second';
    // build continuous button
    secondBox.appendChild(this.buildRadioButton('continue', 'cont', true, idx));
    const labelCont = document.createElement('label');
    labelCont.setAttribute('for', 'cont' + idx.toString());
    labelCont.innerHTML = 'Continue';
    secondBox.appendChild(labelCont);

    // build categorical
    secondBox.appendChild(this.buildRadioButton('categoriaal', 'cate', false, idx));
    const labelCate = document.createElement('label');
    labelCate.setAttribute('for', 'cont' + idx.toString());
    labelCate.innerHTML = 'Categoriaal';
    secondBox.appendChild(labelCate);
    secondBox.append(document.createElement('br'));



    return secondBox;
  }
// Build the "form-control" element
// Contains a text field containing the name of the column
  buildFormControl(colName: string, idx: number): Element {
    const formControl = document.createElement('input');
    formControl.setAttribute('type', 'text');
    formControl.disabled = true;
    formControl.className = 'form-control';
    formControl.placeholder = colName;
    formControl.id = 'rword' + idx.toString();

    return formControl;
  }
// Build n "form-box-first", "form-box-second" and "form-control" elements and
// append to parent. n being the length of columns.
// Used for setting parameters per column
  buildBoxes(columns: string[], parent: Element): void {
    for (let i = 0; i < columns.length; i++) {
      const colName = columns[i];

      // build the first box (input/output)
      parent.appendChild(this.buildFirstBox(i));

      // build the second box (continue/categoriaal)
      parent.appendChild(this.buildSecondBox(i));

      // build the form control box (with column label)
      parent.appendChild(this.buildFormControl(colName, i));

    }
  }
// Function that starts after initialising the component.
// Fill the ".first-part" div with elements
// See buildBoxes and the functions it calls for more
  ngOnInit(): void {
    const columns = sessionStorage.getItem('words').split(',');
    const parent = document.querySelector('.first-part');
    // build the input box for every column
    this.buildBoxes(columns, parent);
  }

  // Query all of the "form-box-first", "form-box-second" and "form-control" elements
  // Saves the resultant settings to sessionStorage.
  // sessionStorage('input_types') contains continuous/categorical per column
  // sessionStorage('col_types') contains input/output/exclude per column
  queryFunc() {
    const form = document.querySelector('.form-box');
    const firstPart = form.querySelector('.first-part');
    const firstBoxes = firstPart.querySelectorAll('.form-box-first');
    const secondBoxes = firstPart.querySelectorAll('.form-box-second');
    const formControls = firstPart.querySelectorAll('.form-control');

    const inputTypes = [];
    const colTypes = [];
    let outputCount = 0;
    for (let i = 0; i < firstBoxes.length; i++) {
      const isInput = ((firstBoxes.item(i)[0]) as HTMLInputElement).checked;
      const isOutput = ((firstBoxes.item(i)[1]) as HTMLInputElement).checked;
      if (isInput === true) {
        colTypes.push('input');
      } else if (isOutput === true) {
        colTypes.push('output');
        outputCount++;
        // If more than one output, reload
        if (outputCount > 1) {
          alert('Je mag maar één kolom markeren als output.');
          window.location.reload();
        }
      } else {
        colTypes.push('exclude');
      }
      const isCont = ((secondBoxes.item(i)[0]) as HTMLInputElement).checked;
      if (isCont === true) {
        inputTypes.push('continue');
      } else {
        inputTypes.push('categoriaal');
      }
    }
    sessionStorage.setItem('inputTypes', inputTypes.toString());
    sessionStorage.setItem('col_types', colTypes.toString());
  }

  // Saves column settings to session storage, then checks if field "name" has been filled out
  // If not, reload page
  // Else, set name and continue to dmn-page
  myFunction() {
    this.queryFunc();
    const value1 = (document.getElementById('title') as HTMLInputElement).value;

    if (value1 === '') {
      alert('Name must be filled out');
      location.reload();
    }
    sessionStorage.setItem('name', value1);
  }
// Function for popup with question mark
  myPopup() {
    const popup = document.getElementById('myPopup');
    popup.classList.toggle('show');
  }


}

