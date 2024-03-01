import { Component } from '@angular/core'

import { MatDialogContent } from '@angular/material/dialog'
import { MatDialogRef } from '@angular/material/dialog'
import { MatDialogTitle } from '@angular/material/dialog'
import { MatGridListModule } from '@angular/material/grid-list'

@Component({
    selector: 'app-entry-page',
    standalone: true,
    imports: [MatDialogTitle, MatDialogContent, MatGridListModule],
    templateUrl: './entry-page.component.html',
    styleUrl: './entry-page.component.scss',
})
export class EntryPageComponent {
    constructor(public dialogRef: MatDialogRef<EntryPageComponent>) {}

    selectOption(option: string) {
        this.dialogRef.close(option)
    }
}
