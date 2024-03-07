import { Component, OnInit } from '@angular/core'

import { CommonModule } from '@angular/common'
import { MatGridListModule } from '@angular/material/grid-list'
import { MatTabsModule } from '@angular/material/tabs'
import { MatProgressBarModule } from '@angular/material/progress-bar'

import { MatIconModule } from '@angular/material/icon'
import { MatDividerModule } from '@angular/material/divider'
import { MatButtonModule } from '@angular/material/button'

import { ScatterPlotComponent } from './scatter-plot/scatter-plot.component'
import { LinePlotsComponent } from './line-plots/line-plots.component'

import { InteractionsService } from './interactions.service'
import { HttpService } from './http.service'

import * as d3 from 'd3'

@Component({
    selector: 'app-visualization',
    standalone: true,
    imports: [
        ScatterPlotComponent,
        LinePlotsComponent,
        CommonModule,
        MatGridListModule,
        MatTabsModule,
        MatProgressBarModule,
        MatIconModule,
        MatDividerModule,
        MatButtonModule,
    ],
    templateUrl: './visualization.component.html',
    styleUrl: './visualization.component.scss',
})
export class VisualizationComponent implements OnInit {
    private dataset = 'resnet-ecg5000'

    private stage = 'train'

    public isRequestPending = true

    projectedTimeSeries = [
        'Projected Time Series',
        'http://localhost:8000/api/get_projected_time_series/?stage=' + this.stage,
        'http://localhost:8000/api/get_projected_time_series_density/?stage=' + this.stage,
        'http://localhost:8000/api/project_time_series/?stage=' + this.stage,
        'http://localhost:8000/api/inverse_project_time_series/?stage=' + this.stage,
    ]
    projectedActivations = [
        'Projected Activations',
        'http://localhost:8000/api/get_projected_activations/?stage=' + this.stage,
        'http://localhost:8000/api/get_projected_activations_density/?stage=' + this.stage,
        'http://localhost:8000/api/project_activations/?stage=' + this.stage,
        'http://localhost:8000/api/inverse_project_activations/?stage=' + this.stage,
    ]
    projectedAttributions = [
        'Projected Attributions',
        'http://localhost:8000/api/get_projected_attributions/?stage=' + this.stage,
        'http://localhost:8000/api/get_projected_attributions_density/?stage=' + this.stage,
        'http://localhost:8000/api/project_attributions/?stage=' + this.stage,
        'http://localhost:8000/api/inverse_project_attributions/?stage=' + this.stage,
    ]

    originalTimeSeries = 'http://localhost:8000/api/get_time_series/?stage=' + this.stage

    constructor(private httpService: HttpService, private interactionsService: InteractionsService) {}

    firstDataLoaded = false

    ngOnInit(): void {
        this.httpService.get<any>(this.originalTimeSeries).subscribe((data: any) => {
            const parsed_data = JSON.parse(data)
            this.interactionsService.setData(parsed_data['data'])

            this.firstDataLoaded = true
            console.log('Data Loaded')
        })

        this.httpService.isRequestPending().subscribe((pending: boolean) => {
            this.isRequestPending = pending
        })
    }

    setColorMode(mode: number = 2) {
        this.interactionsService.setColorMode(mode)
    }
}
