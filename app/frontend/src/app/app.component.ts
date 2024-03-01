import { Component, OnInit } from '@angular/core'
import { RouterOutlet } from '@angular/router'
import { VisualizationComponent } from './visualization/visualization.component'

import { MatDialog } from '@angular/material/dialog'
import { MatDialogModule } from '@angular/material/dialog'
import { EntryPageComponent } from './entry-page/entry-page.component'

@Component({
    selector: 'app-root',
    standalone: true,
    imports: [RouterOutlet, VisualizationComponent, MatDialogModule, EntryPageComponent],
    templateUrl: './app.component.html',
    styleUrl: './app.component.scss',
})
export class AppComponent implements OnInit {
    title = 'Interactive Generation of Counterfactuals for Time Series'

    constructor(public dialog: MatDialog) {}

    ngOnInit() {
        // this.openOptionsModal()
    }

    openOptionsModal(): void {
        const dialogRef = this.dialog.open(EntryPageComponent, {
            width: '98%',
            height: '98%',
            maxWidth: '100%',
            panelClass: 'full-screen-modal',
        })

        dialogRef.afterClosed().subscribe((result) => {
            console.log('The dialog was closed. Selected option:', result)
            // Handle the selected option
        })
    }
}
