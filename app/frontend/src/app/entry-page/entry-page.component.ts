import { Component } from '@angular/core'

import { MatDialog } from '@angular/material/dialog'

@Component({
    selector: 'app-entry-page',
    standalone: true,
    imports: [],
    templateUrl: './entry-page.component.html',
    styleUrl: './entry-page.component.scss',
})
export class EntryPageComponent {
    constructor(public dialog: MatDialog) {}
}
