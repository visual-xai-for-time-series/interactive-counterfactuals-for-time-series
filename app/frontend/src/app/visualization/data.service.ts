import { Injectable } from '@angular/core'

@Injectable({
    providedIn: 'root',
})
export class DataService {
    private counter: number = 0
    private colorScale: any[] = []

    constructor() {}

    getCounter(): any {
        return this.counter
    }

    addToCounter(add: number = 1): void {
        this.counter += add
    }

    setColorScale(newColorScale: any[]): void {
        this.colorScale = newColorScale
    }

    getColorScale(): any[] {
        return this.colorScale
    }
}
