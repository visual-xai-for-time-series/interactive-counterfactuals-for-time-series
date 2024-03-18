import { Component, ElementRef, Input, ViewChild } from '@angular/core'
import { InteractionsService } from '../../interactions.service'

import { MatIconModule } from '@angular/material/icon'
import { MatDividerModule } from '@angular/material/divider'
import { MatButtonModule } from '@angular/material/button'

import * as d3 from 'd3'

@Component({
    selector: 'app-single-line-plot',
    standalone: true,
    imports: [MatIconModule, MatDividerModule, MatButtonModule],
    templateUrl: './single-line-plot.component.html',
    styleUrl: './single-line-plot.component.scss',
})
export class SingleLinePlotComponent {
    @ViewChild('div') divElement: ElementRef<HTMLDivElement> | undefined

    @Input() width
    @Input() height

    private svg: any
    private vis: any

    private xScale: any
    private yScale: any

    private line: any
    private lineData: any = [0, 0, 0, 0]
    private linePlotted: boolean = false
    private lineModified: boolean = false

    constructor(private interactionsService: InteractionsService) {}

    ngAfterViewInit(): void {
        if (typeof this.svg === 'undefined') {
            this.createLinePlot()
        }

        this.interactionsService.getLineData.subscribe((data) => {
            if (data != null && !this.linePlotted) {
                this.addLine(data)
                this.linePlotted = true
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
        this.lineData = data_with_labels

        const idx = this.lineData[0]
        const data = this.lineData[1]
        const label = this.lineData[2]
        const prediction = this.lineData[3]

        const backgroundLine = this.vis
            .append('path')
            .datum(data)
            .attr('class', 'background-data-line')
            .attr('data-idx', idx)
            .attr('fill', 'none')
            .attr('stroke', 'gray')
            .attr('opacity', 0.5)
            .attr('stroke-width', 2)
            .attr('base-stroke', 'gray')
            .attr('d', this.line)

        const pointLine = this.vis
            .append('path')
            .datum(data)
            .attr('class', 'data-line')
            .attr('data-idx', idx)
            .attr('fill', 'none')
            .attr('stroke', d3.schemeDark2[label])
            .attr('stroke-width', 2)
            .attr('base-stroke', d3.schemeDark2[label])
            .attr('d', this.line)
            .on('mouseover', (d: any) => {
                const idx = d3.select(d.target).attr('data-idx')
                this.interactionsService.setHighlightData(idx)
            })
            .on('mouseout', () => {
                this.interactionsService.setHighlightData(null)
            })

        function dragstarted(this: any, d) {
            d3.select(this).raise().classed('active', true)
        }

        const that = this
        function dragged(this: any, d: any) {
            const extent = d3.extent(that.yScale.range())

            if (d.y > extent[0] && d.y < extent[1]) {
                that.lineData[1][parseInt(d3.select(this).attr('tp'))] = that.yScale.invert(d.y)
                // data[parseInt(xPos)] = that.yScale.invert(d.y)
                that.lineModified = true
            }

            d3.select(this).attr('cy', d.y)
        }

        function dragended(this: any, d) {
            d3.select(this).classed('active', false)
            pointLine.attr('d', that.line)
        }

        const drag = d3.drag().on('start', dragstarted).on('drag', dragged).on('end', dragended)

        this.vis
            .selectAll('.line-circle')
            .data(data)
            .enter()
            .append('circle')
            .attr('class', 'line-circle')
            .attr('r', 3)
            .attr('tp', (_, i) => {
                return i
            })
            .attr('cx', (_, i) => {
                return this.xScale(i)
            })
            .attr('cy', (d) => {
                return this.yScale(d)
            })
            .call(drag)
    }

    private highlightData(filterData: any) {
        if (this.svg) {
            this.svg
                .selectAll('.data-line')
                .filter(function (this: any) {
                    return d3.select(this).attr('data-idx') === filterData
                })
                .style('stroke-width', 4)
        }
    }

    private unHighlightData() {
        if (this.svg) {
            this.svg.selectAll('.data-line').style('stroke-width', null)
        }
    }

    public removeLine(): void {
        this.lineData = [0, 0, 0, 0]
        this.linePlotted = false
        this.lineModified = false

        this.svg.selectAll('.data-line').remove()
        this.svg.selectAll('.line-circle').remove()
        this.svg.selectAll('.background-data-line').remove()

        this.interactionsService.clearScatterData()
    }

    public project() {
        if (this.lineModified) {
            this.lineData[0] = `icf-${this.lineData[0]}`
            const orgIdx = this.svg.select('.data-line').attr('data-idx')
            this.svg.select('.data-line').attr('org-data-idx', orgIdx).attr('data-idx', this.lineData[0])

            this.interactionsService.addNewScatterData(this.lineData)
        }
    }
}
