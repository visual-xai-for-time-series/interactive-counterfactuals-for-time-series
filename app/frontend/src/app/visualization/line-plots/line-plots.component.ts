import { Component, ElementRef, Input, ViewChild } from '@angular/core'
import { CommonModule } from '@angular/common'

import { MultiLinePlotComponent } from './multi-line-plot/multi-line-plot.component'
import { SingleLinePlotComponent } from './single-line-plot/single-line-plot.component'

import { MatGridListModule } from '@angular/material/grid-list'
import { MatTabsModule } from '@angular/material/tabs'

@Component({
    selector: 'app-line-plots',
    standalone: true,
    imports: [CommonModule, MultiLinePlotComponent, SingleLinePlotComponent, MatGridListModule, MatTabsModule],
    templateUrl: './line-plots.component.html',
    styleUrl: './line-plots.component.scss',
})
export class LinePlotsComponent {
    @ViewChild('element') element: ElementRef<HTMLDivElement> | undefined

    public width: number | null = null
    public height: number | null = null

    constructor() {}

    ngAfterViewInit(): void {
        if (typeof this.element != 'undefined') {
            this.width = this.element.nativeElement.offsetWidth
            this.height = this.element.nativeElement.offsetHeight
        }
    }
}
