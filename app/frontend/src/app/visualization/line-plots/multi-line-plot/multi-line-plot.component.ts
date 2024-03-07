import { Component, ElementRef, Input, ViewChild } from '@angular/core'
import { InteractionsService } from '../../interactions.service'

import { MatIconModule } from '@angular/material/icon'
import { MatDividerModule } from '@angular/material/divider'
import { MatButtonModule } from '@angular/material/button'

import * as d3 from 'd3'

@Component({
    selector: 'app-multi-line-plot',
    standalone: true,
    imports: [MatIconModule, MatDividerModule, MatButtonModule],
    templateUrl: './multi-line-plot.component.html',
    styleUrl: './multi-line-plot.component.scss',
})
export class MultiLinePlotComponent {
    @ViewChild('div') divElement: ElementRef<HTMLDivElement> | undefined

    @Input() width
    @Input() height

    private svg: any
    private vis: any

    private xScale: any
    private yScale: any

    private line: any
    private line_stroke_width: number = 3

    private renderedInstances: string[] = []

    constructor(private interactionsService: InteractionsService) {}

    ngAfterViewInit(): void {
        if (typeof this.svg === 'undefined') {
            this.createLinePlot()
        }

        this.interactionsService.getLineData.subscribe((data) => {
            if (data != null) {
                this.addLine(data)
            }
        })

        this.interactionsService.getHighlightData.subscribe((data) => {
            if (data != null) {
                this.highlightData(data)
            } else {
                this.unHighlightData()
            }
        })
    }

    private createLinePlot(): void {
        const dataWithLabels = this.interactionsService.getData()
        const data = dataWithLabels[0]

        const containerWidth = this.width - 20
        const containerHeight = this.height - 140

        const margin = { top: 20, right: 20, bottom: 50, left: 70 }
        const width = containerWidth - margin.left - margin.right
        const height = containerHeight - margin.top - margin.bottom

        this.svg = d3
            .select(this.divElement?.nativeElement)
            .append('svg')
            .attr('width', containerWidth)
            .attr('height', containerHeight)
        this.vis = this.svg.append('g').attr('transform', 'translate(' + margin.left + ',' + margin.top + ')')

        const length = data[0].length
        this.xScale = d3.scaleLinear().domain([0, length]).range([0, width])

        const extent = d3.extent(data.flat(), (d) => d)
        this.yScale = d3.scaleLinear().domain(extent).range([height, 0])

        const line = d3
            .line()
            .x((_, i: number) => this.xScale(i))
            .y((d) => this.yScale(d))

        this.line = line

        this.vis
            .append('g')
            .attr('transform', 'translate(0,' + height + ')')
            .call(d3.axisBottom(this.xScale))

        this.vis.append('g').call(d3.axisLeft(this.yScale))

        this.vis
            .append('text')
            .attr('transform', 'translate(' + width / 2 + ' ,' + (height + margin.top + 20) + ')')
            .style('text-anchor', 'middle')
            .text('time')

        this.vis
            .append('text')
            .attr('transform', 'rotate(-90)')
            .attr('y', 0 - margin.left)
            .attr('x', 0 - height / 2)
            .attr('dy', '1em')
            .style('text-anchor', 'middle')
            .text('value')
    }

    private addLine(data_with_labels: any[]): void {
        console.log(data_with_labels)

        const idx = data_with_labels[0]
        const data = data_with_labels[1]
        const label = data_with_labels[2]
        const prediction = data_with_labels[3]

        if (!(idx in this.renderedInstances)) {
            this.vis
                .append('path')
                .datum(data)
                .attr('class', 'data-line')
                .attr('data-idx', idx)
                .attr('fill', 'none')
                .attr('stroke', d3.schemeDark2[prediction])
                .attr('stroke-width', this.line_stroke_width)
                .attr('base-stroke', d3.schemeDark2[prediction])
                .attr('d', this.line)
                .on('mouseover', (d: any) => {
                    const idx = d3.select(d.target).attr('data-idx')
                    this.interactionsService.setHighlightData(idx)
                })
                .on('mouseout', () => {
                    this.interactionsService.setHighlightData(null)
                })

            this.renderedInstances.push(idx)
        }
    }

    highlightedLine: any = null
    private highlightData(filterData: any) {
        if (this.svg) {
            const toHighlight = this.svg.selectAll('.data-line').filter(function (this: any) {
                return d3.select(this).attr('data-idx') === filterData
            })

            if (this.highlightedLine == null) {
                this.highlightedLine = toHighlight.clone()
                this.highlightedLine.style('stroke', 'black').style('stroke-width', 8)
            }
            toHighlight.style('stroke-width', 4).raise()
        }
    }

    private unHighlightData() {
        if (this.svg) {
            this.svg.selectAll('.data-line').style('stroke-width', null)
            if (this.highlightedLine) {
                this.highlightedLine.remove()
                this.highlightedLine = null
            }
        }
    }

    public removeLines(): void {
        this.renderedInstances = []
        this.svg.selectAll('.data-line').remove()

        this.interactionsService.clearScatterData()
    }
}
